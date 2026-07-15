"""Tear down the per-run Azure resources for one QE VM.

Deletes, in dependency order: VM -> NIC -> public IP -> OS disk.
Each resource is checked by name first, so this is safe to call after a
PARTIAL provisioning failure (e.g. public IP created but NIC attach failed)
and safe to call twice. The resource group, VNet, subnet and NSG are
persistent and are deliberately never deleted here.

CLI use (e.g. to clean up after a crash):
    .venv/bin/python azure/teardown_vm.py qe-20260702-153000-ab12
"""

import sys

from azure.core.exceptions import ResourceNotFoundError

from provision_vm import get_clients, load_config, resource_names


def _delete(kind, name, get_fn, delete_fn, deleted, errors, log):
    """Delete one resource if it exists; record its ID or the failure."""
    try:
        resource = get_fn(name)
    except ResourceNotFoundError:
        log(f"{kind} {name}: not found, nothing to delete.")
        return
    try:
        log(f"Deleting {kind} {name}...")
        delete_fn(name).result()
        deleted.append(resource.id)
        log(f"Deleted {resource.id}")
    except Exception as exc:  # keep going — delete as much as possible
        errors.append(f"{kind} {name}: {exc}")
        log(f"FAILED to delete {kind} {name}: {exc}")


def teardown_vm(vm_name, cfg=None, log=print):
    """Delete the VM and its NIC, public IP and OS disk. Returns the list of
    deleted resource IDs. Raises RuntimeError if any deletion failed, after
    attempting all of them."""
    cfg = cfg or load_config()
    group = cfg["AZURE_GROUP"]
    names = resource_names(vm_name)
    _, network_client, compute_client = get_clients(cfg)

    deleted, errors = [], []

    # The VM may have been created with a non-deterministic disk name in an
    # older run; prefer the name recorded on the VM itself when available.
    os_disk_name = names["os_disk"]
    try:
        vm = compute_client.virtual_machines.get(group, names["vm"])
        if vm.storage_profile and vm.storage_profile.os_disk:
            os_disk_name = vm.storage_profile.os_disk.name or os_disk_name
    except ResourceNotFoundError:
        pass

    _delete(
        "VM",
        names["vm"],
        lambda n: compute_client.virtual_machines.get(group, n),
        lambda n: compute_client.virtual_machines.begin_delete(group, n),
        deleted, errors, log,
    )
    _delete(
        "NIC",
        names["nic"],
        lambda n: network_client.network_interfaces.get(group, n),
        lambda n: network_client.network_interfaces.begin_delete(group, n),
        deleted, errors, log,
    )
    _delete(
        "public IP",
        names["public_ip"],
        lambda n: network_client.public_ip_addresses.get(group, n),
        lambda n: network_client.public_ip_addresses.begin_delete(group, n),
        deleted, errors, log,
    )
    _delete(
        "OS disk",
        os_disk_name,
        lambda n: compute_client.disks.get(group, n),
        lambda n: compute_client.disks.begin_delete(group, n),
        deleted, errors, log,
    )

    if errors:
        raise RuntimeError(
            "Teardown incomplete — delete the remaining resources manually "
            "(portal or `az resource delete`): " + "; ".join(errors)
        )
    return deleted


def main():
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <vm_name>", file=sys.stderr)
        sys.exit(2)
    deleted = teardown_vm(sys.argv[1])
    print(f"Teardown complete. {len(deleted)} resource(s) deleted.")


if __name__ == "__main__":
    main()
