#!/usr/bin/env python3
"""Stateless status poll for a detached QE job on a named VM.

Reads the per-command bookkeeping files that run_job.py's detached wrapper
maintains in ~/qe_job (<base>.cmd/.pid/.exit) and prints one line per
command slot: DONE(<status>), RUNNING, or ABSENT. Exits 0 if every
expected command slot is DONE with status 0, 2 if any slot failed, 1
otherwise (still running / not started).

Usage: python poll_job.py <vm-name> <n-commands>
"""

from __future__ import annotations

import sys

from provision_vm import get_clients, load_config, open_ssh, resource_names, run_quiet


def main() -> int:
    vm_name, n_commands = sys.argv[1], int(sys.argv[2])
    cfg = load_config()
    _, network_client, _ = get_clients(cfg)
    names = resource_names(vm_name)
    ip = network_client.public_ip_addresses.get(
        cfg["AZURE_GROUP"], names["public_ip"]).ip_address
    if not ip:
        print(f"{vm_name}: no public IP (VM gone?)")
        return 1
    ssh = open_ssh(ip)
    try:
        all_done, any_failed = True, False
        for i in range(1, n_commands + 1):
            base = f"run_stdout_{i}"
            code, exit_txt = run_quiet(ssh, f"cat qe_job/{base}.exit 2>/dev/null")
            if code == 0 and exit_txt.strip():
                status = int(exit_txt.strip())
                print(f"cmd {i}: DONE({status})")
                any_failed |= status != 0
                continue
            all_done = False
            code, pid_txt = run_quiet(ssh, f"cat qe_job/{base}.pid 2>/dev/null")
            if code == 0 and pid_txt.strip():
                alive, _ = run_quiet(ssh, f"ps -p {pid_txt.strip()} > /dev/null 2>&1; echo $?")
                running = alive == 0 and _.strip() == "0"
                print(f"cmd {i}: {'RUNNING' if running else 'STALLED (pid dead, no exit)'}")
            else:
                print(f"cmd {i}: ABSENT")
        if any_failed:
            return 2
        return 0 if all_done else 1
    finally:
        ssh.close()


if __name__ == "__main__":
    sys.exit(main())
