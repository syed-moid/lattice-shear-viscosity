"""Provision an Azure VM with Quantum ESPRESSO installed at boot via cloud-init.

Resource model:
  Persistent (created once if absent, NEVER deleted by teardown_vm.py):
    resource group (AZURE_GROUP), virtual network, subnet, network security group.
  Per-run (deterministic names derived from the VM name, all deleted by
  teardown_vm.py): VM, NIC, public IP, OS disk.

Credentials come from .env at the repo root (loaded with python-dotenv).
The key names there are AZURE_SUBSCRIPTION_ID / AZURE_TENANT_ID / AZURE_GROUP /
LOCATION / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET, which do not match
EnvironmentCredential's defaults — so the ClientSecretCredential is built
explicitly from them.
"""

import argparse
import base64
import os
import socket
import sys
import time
from pathlib import Path

import paramiko
from azure.identity import ClientSecretCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute import models as compute_models
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.network import models as network_models
# azure-mgmt-resource >= 26 moved the client into the .resources subpackage
from azure.mgmt.resource.resources import ResourceManagementClient
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

VNET_NAME = "qe-vnet"
SUBNET_NAME = "qe-subnet"
NSG_NAME = "qe-nsg"
VNET_ADDRESS_SPACE = "10.42.0.0/16"
SUBNET_PREFIX = "10.42.0.0/24"

ADMIN_USERNAME = "azureuser"
# FSv2 is the family the East US quota increase was approved for
# (FSv2 Family + Total Regional vCPUs, both 32) — keep the default in that family.
DEFAULT_VM_SIZE = "Standard_F8s_v2"

SSH_WAIT_TIMEOUT_S = 600        # port 22 reachable
QE_WAIT_TIMEOUT_S = 1500        # cloud-init apt install finished
POLL_INTERVAL_S = 15

UBUNTU_IMAGE = {
    "publisher": "Canonical",
    "offer": "0001-com-ubuntu-server-jammy",
    "sku": "22_04-lts-gen2",
    "version": "latest",
}

# QE is installed in the background after first boot; run_job.py polls
# `command -v pw.x` over SSH until this has finished.
CLOUD_INIT = """\
#cloud-config
package_update: true
packages:
  - quantum-espresso
"""

REQUIRED_ENV = [
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_TENANT_ID",
    "AZURE_GROUP",
    "LOCATION",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
]


def load_config():
    """Read the required Azure settings from the environment (.env)."""
    cfg = {name: os.environ.get(name, "").strip() for name in REQUIRED_ENV}
    missing = [name for name, value in cfg.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required .env entries: {', '.join(missing)} "
            f"(expected in {REPO_ROOT / '.env'})"
        )
    return cfg


def get_credential(cfg):
    return ClientSecretCredential(
        tenant_id=cfg["AZURE_TENANT_ID"],
        client_id=cfg["AZURE_CLIENT_ID"],
        client_secret=cfg["AZURE_CLIENT_SECRET"],
    )


def get_clients(cfg, credential=None):
    credential = credential or get_credential(cfg)
    sub = cfg["AZURE_SUBSCRIPTION_ID"]
    return (
        ResourceManagementClient(credential, sub),
        NetworkManagementClient(credential, sub),
        ComputeManagementClient(credential, sub),
    )


def resource_names(vm_name):
    """Deterministic per-run resource names, shared with teardown_vm.py so
    partial provisioning failures can still be cleaned up by name."""
    return {
        "vm": vm_name,
        "nic": f"{vm_name}-nic",
        "public_ip": f"{vm_name}-ip",
        "os_disk": f"{vm_name}-osdisk",
    }


def find_ssh_public_key():
    override = os.environ.get("SSH_PUBLIC_KEY_PATH", "").strip()
    if override:
        candidates = [Path(override).expanduser()]
    else:
        candidates = [
            Path.home() / ".ssh" / "id_ed25519.pub",
            Path.home() / ".ssh" / "id_rsa.pub",
        ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        "No SSH public key found (tried "
        + ", ".join(str(p) for p in candidates)
        + "). Run `ssh-keygen -t ed25519` or set SSH_PUBLIC_KEY_PATH in .env."
    )


def find_ssh_private_key():
    pub = find_ssh_public_key()
    priv = pub.with_suffix("")
    if not priv.is_file():
        raise RuntimeError(f"Private key {priv} matching {pub} not found.")
    return priv


def _ensure_network(network_client, group, location, log):
    """Get-or-create the persistent VNet/subnet/NSG. Never torn down."""
    try:
        network_client.virtual_networks.get(group, VNET_NAME)
        log(f"VNet {VNET_NAME} already exists.")
    except Exception:
        log(f"Creating VNet {VNET_NAME} ({VNET_ADDRESS_SPACE})...")
        # NOTE: model classes, not raw dicts — the new-generation network/compute
        # SDKs send dicts verbatim as JSON, so snake_case dict keys are rejected
        # by the API (learned from a real failed run).
        network_client.virtual_networks.begin_create_or_update(
            group,
            VNET_NAME,
            network_models.VirtualNetwork(
                location=location,
                address_space=network_models.AddressSpace(
                    address_prefixes=[VNET_ADDRESS_SPACE]),
                subnets=[network_models.Subnet(
                    name=SUBNET_NAME, address_prefix=SUBNET_PREFIX)],
            ),
        ).result()
    subnet = network_client.subnets.get(group, VNET_NAME, SUBNET_NAME)

    try:
        nsg = network_client.network_security_groups.get(group, NSG_NAME)
        log(f"NSG {NSG_NAME} already exists.")
    except Exception:
        log(f"Creating NSG {NSG_NAME} (inbound SSH only)...")
        nsg = network_client.network_security_groups.begin_create_or_update(
            group,
            NSG_NAME,
            network_models.NetworkSecurityGroup(
                location=location,
                security_rules=[network_models.SecurityRule(
                    name="allow-ssh",
                    priority=1000,
                    direction="Inbound",
                    access="Allow",
                    protocol="Tcp",
                    source_address_prefix="*",
                    source_port_range="*",
                    destination_address_prefix="*",
                    destination_port_range="22",
                )],
            ),
        ).result()
    return subnet, nsg


def provision_vm(vm_name, vm_size=DEFAULT_VM_SIZE, cfg=None, log=print):
    """Create public IP -> NIC -> VM (with QE cloud-init). Returns the VM's
    public IP address as a string."""
    cfg = cfg or load_config()
    group, location = cfg["AZURE_GROUP"], cfg["LOCATION"]
    names = resource_names(vm_name)
    ssh_pubkey = find_ssh_public_key().read_text().strip()

    resource_client, network_client, compute_client = get_clients(cfg)

    log(f"Ensuring resource group {group} in {location}...")
    resource_client.resource_groups.create_or_update(group, {"location": location})

    subnet, nsg = _ensure_network(network_client, group, location, log)

    log(f"Creating public IP {names['public_ip']}...")
    public_ip = network_client.public_ip_addresses.begin_create_or_update(
        group,
        names["public_ip"],
        network_models.PublicIPAddress(
            location=location,
            sku=network_models.PublicIPAddressSku(name="Standard"),
            public_ip_allocation_method="Static",
        ),
    ).result()

    log(f"Creating NIC {names['nic']}...")
    nic = network_client.network_interfaces.begin_create_or_update(
        group,
        names["nic"],
        network_models.NetworkInterface(
            location=location,
            network_security_group=network_models.NetworkSecurityGroup(id=nsg.id),
            ip_configurations=[network_models.NetworkInterfaceIPConfiguration(
                name="primary",
                subnet=network_models.Subnet(id=subnet.id),
                public_ip_address=network_models.PublicIPAddress(id=public_ip.id),
            )],
        ),
    ).result()

    log(f"Creating VM {vm_name} ({vm_size}) with QE cloud-init...")
    compute_client.virtual_machines.begin_create_or_update(
        group,
        vm_name,
        compute_models.VirtualMachine(
            location=location,
            hardware_profile=compute_models.HardwareProfile(vm_size=vm_size),
            storage_profile=compute_models.StorageProfile(
                image_reference=compute_models.ImageReference(**UBUNTU_IMAGE),
                os_disk=compute_models.OSDisk(
                    name=names["os_disk"],
                    create_option="FromImage",
                    # Backstop: if the teardown script itself dies after the VM
                    # delete, the disk/NIC still go with the VM.
                    delete_option="Delete",
                    managed_disk=compute_models.ManagedDiskParameters(
                        storage_account_type="StandardSSD_LRS"),
                ),
            ),
            os_profile=compute_models.OSProfile(
                computer_name=vm_name,
                admin_username=ADMIN_USERNAME,
                custom_data=base64.b64encode(CLOUD_INIT.encode()).decode(),
                linux_configuration=compute_models.LinuxConfiguration(
                    disable_password_authentication=True,
                    ssh=compute_models.SshConfiguration(
                        public_keys=[compute_models.SshPublicKey(
                            path=f"/home/{ADMIN_USERNAME}/.ssh/authorized_keys",
                            key_data=ssh_pubkey,
                        )]
                    ),
                ),
            ),
            network_profile=compute_models.NetworkProfile(
                network_interfaces=[compute_models.NetworkInterfaceReference(
                    id=nic.id, delete_option="Delete")],
            ),
        ),
    ).result()

    ip_address = network_client.public_ip_addresses.get(
        group, names["public_ip"]
    ).ip_address
    log(f"VM {vm_name} is running at {ip_address}.")
    return ip_address


# --- SSH / QE readiness (used both here in standalone mode and by run_job.py) ---


def wait_for_ssh_port(ip, log, timeout=SSH_WAIT_TIMEOUT_S):
    log(f"Waiting for SSH port 22 on {ip} (timeout {timeout}s)...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ip, 22), timeout=10):
                log("SSH port is open.")
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"SSH port 22 on {ip} not reachable after {timeout}s")


def open_ssh(ip):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        ip,
        username=ADMIN_USERNAME,
        key_filename=str(find_ssh_private_key()),
        timeout=30,
    )
    return client


def run_quiet(ssh, command):
    """Run a remote command, return (exit_status, stdout_text)."""
    _, stdout, _ = ssh.exec_command(command)
    text = stdout.read().decode(errors="replace").strip()
    return stdout.channel.recv_exit_status(), text


def wait_for_qe_ready(ip, log, timeout=QE_WAIT_TIMEOUT_S):
    """SSH must accept a session AND `pw.x` AND `mpirun` must exist —
    cloud-init installs QE in the background after boot, so SSH-up alone is
    not 'ready', and the standard run commands use `mpirun -np <vCPUs/2>`,
    so a VM missing OpenMPI must fail fast here, not 20 minutes into a job.
    Returns the installed quantum-espresso package version."""
    wait_for_ssh_port(ip, log)
    log(f"Waiting for cloud-init to finish installing QE (timeout {timeout}s)...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ssh = open_ssh(ip)
        except Exception as exc:
            log(f"SSH not accepting sessions yet ({exc}); retrying...")
            time.sleep(POLL_INTERVAL_S)
            continue
        try:
            status, _ = run_quiet(ssh, "command -v pw.x && command -v mpirun")
            if status == 0:
                _, version = run_quiet(
                    ssh,
                    "dpkg-query -W -f='${Version}' quantum-espresso 2>/dev/null",
                )
                log(f"pw.x and mpirun are available (quantum-espresso {version or 'unknown'}).")
                return version or "unknown"
            status, ci = run_quiet(ssh, "cloud-init status 2>/dev/null")
            if "error" in ci:
                raise RuntimeError(f"cloud-init failed on the VM: {ci!r}")
            log(f"pw.x/mpirun not present yet ({ci or 'cloud-init running'}); polling...")
        finally:
            ssh.close()
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"QE not installed after {timeout}s — cloud-init too slow or failed")


def print_connection_info(vm_name, vm_size, region, ip, log=print):
    """Connection block for manual SSH access (e.g. tailing logs mid-job)."""
    key = find_ssh_private_key()
    for line in (
        "=" * 58,
        "VM CONNECTION INFO",
        f"  VM name:     {vm_name}",
        f"  Size:        {vm_size}",
        f"  Region:      {region}",
        f"  Public IP:   {ip}",
        f"  SSH user:    {ADMIN_USERNAME}",
        f"  Connect:     ssh {ADMIN_USERNAME}@{ip}",
        f"  Private key: {key}",
        "=" * 58,
    ):
        log(line)


def main():
    parser = argparse.ArgumentParser(
        description="Provision a QE-ready Azure VM for MANUAL use and leave it "
                    "running (no teardown — use run_job.py for automated jobs)."
    )
    parser.add_argument("--vm-name", required=True)
    parser.add_argument("--size", default=DEFAULT_VM_SIZE)
    args = parser.parse_args()

    cfg = load_config()
    try:
        ip = provision_vm(args.vm_name, vm_size=args.size, cfg=cfg)
        wait_for_qe_ready(ip, print)
    except (Exception, KeyboardInterrupt):
        print(
            f"\nProvisioning did not complete — resources may still exist. "
            f"Clean up with:\n    .venv/bin/python azure/teardown_vm.py {args.vm_name}",
            file=sys.stderr,
        )
        raise
    print_connection_info(args.vm_name, args.size, cfg["LOCATION"], ip)
    print(
        "\nWARNING: this VM is LEFT RUNNING and BILLING until you delete it.\n"
        f"When done:  .venv/bin/python azure/teardown_vm.py {args.vm_name}"
    )


if __name__ == "__main__":
    main()
