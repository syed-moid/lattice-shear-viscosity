"""End-to-end QE run on a throwaway Azure VM, with guaranteed teardown.

    .venv/bin/python azure/run_job.py \
        --input-dir qe/SrTiO3/harmonic_gamma \
        --command "mpirun -np 4 pw.x -i scf.in" \
        --command "mpirun -np 4 ph.x -i ph.in"

MPI rule: -np <physical cores> = vCPUs/2 (Fsv2 vCPUs are hyperthreads and
OpenMPI slots default to PHYSICAL cores; more and mpirun refuses to start).
F8s_v2 (default) -> -np 4, F16s_v2 -> -np 8, F32s_v2 -> -np 16. The VM's
CPU topology is logged after provisioning and a WARNING is printed if a
command's -np exceeds the physical core count.

--command may be given multiple times; the commands run IN ORDER over the
same SSH session on the SAME VM. This matters for pw.x -> ph.x chains:
ph.x restarts from pw.x's outdir/<prefix>.save/ on the VM's disk, which a
teardown between commands would destroy. If a command exits non-zero, the
remaining commands are skipped, whatever partial outputs exist are still
downloaded, and then --on-failure decides:
  keep (default): the VM is LEFT RUNNING (and billing) with connection
      info printed, so the failure can be diagnosed in place;
  teardown: unconditional cleanup as for any other error.

ALL input files (scf.in AND ph.in, ...) must already be in --input-dir
before the run: inputs are uploaded once, up front. Pseudopotentials are
uploaded automatically from --pseudo-dir (default qe/pseudopotentials/)
into a pseudo/ subdirectory of the remote job dir, so pseudo_dir='./pseudo/'
in the .in files resolves on the VM.

For every non-command failure — SSH timeout, upload/download error,
Ctrl+C, partial provisioning — teardown still ALWAYS runs, via a `finally`
block (teardown_vm checks each resource by name before deleting).

A provenance log is appended to <input-dir>/run_log.txt: VM name/size/
region, timestamps, QE version, per-command exit statuses, and the
resource IDs confirmed deleted. Copy the relevant lines into
docs/provenance.md after the run.
"""

import argparse
import posixpath
import re
import secrets
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from provision_vm import (
    DEFAULT_VM_SIZE,
    load_config,
    open_ssh,
    print_connection_info,
    provision_vm,
    resource_names,
    run_quiet,
    wait_for_qe_ready,
)
from teardown_vm import teardown_vm

REMOTE_JOB_DIR = "qe_job"
REMOTE_POLL_S = 10


class SshSession:
    """Duck-types the parts of paramiko.SSHClient this script uses, with
    lazy connect and automatic reconnect if the transport died. Lets a
    multi-hour poll survive dropped connections."""

    def __init__(self, ip):
        self.ip = ip
        self._client = None

    def _ensure(self):
        transport = self._client.get_transport() if self._client else None
        if transport is None or not transport.is_active():
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = open_ssh(self.ip)
        return self._client

    def exec_command(self, *args, **kwargs):
        return self._ensure().exec_command(*args, **kwargs)

    def open_sftp(self):
        return self._ensure().open_sftp()

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


class RunLog:
    """Print to console and append to <input-dir>/run_log.txt."""

    def __init__(self, path):
        self.path = Path(path)

    def log(self, message):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a") as fh:
            fh.write(line + "\n")


def check_mpi_slots(ssh, commands, log):
    """Log the VM's CPU topology and WARN (don't abort) if any command asks
    mpirun for more ranks than there are PHYSICAL cores — OpenMPI's slot
    count defaults to physical cores (Fsv2 vCPUs are 2-way hyperthreads),
    and exceeding it makes mpirun refuse to launch."""
    _, cpuinfo = run_quiet(
        ssh, 'lscpu | grep -E "^CPU\\(s\\)|^Thread\\(s\\) per core"')
    log(f"VM CPU topology: {' | '.join(cpuinfo.split(chr(10)))}")
    try:
        fields = dict(
            line.split(":", 1) for line in cpuinfo.splitlines() if ":" in line)
        vcpus = int(fields["CPU(s)"].strip())
        threads_per_core = int(fields["Thread(s) per core"].strip())
        physical = vcpus // threads_per_core
    except (KeyError, ValueError, ZeroDivisionError):
        log("Could not parse CPU topology — skipping the mpirun slot check.")
        return
    for command in commands:
        match = re.search(r"\bmpirun\b.*?-np\s+(\d+)", command)
        if match and int(match.group(1)) > physical:
            log(f"WARNING: {command!r} requests -np {match.group(1)} but the "
                f"VM has only {physical} physical cores ({vcpus} vCPUs / "
                f"{threads_per_core} threads per core) — OpenMPI will refuse "
                f"to launch; use -np {physical} or add --use-hwthread-cpus.")


def upload_inputs(ssh, input_dir, log):
    """Upload every regular file in input_dir to ~/qe_job/. Returns the
    uploaded file names (used later to avoid re-downloading inputs)."""
    files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "run_log.txt"
    )
    if not files:
        raise RuntimeError(f"No input files found in {input_dir}")
    run_quiet(ssh, f"mkdir -p {REMOTE_JOB_DIR}")
    sftp = ssh.open_sftp()
    try:
        for path in files:
            log(f"Uploading {path.name} ({path.stat().st_size} bytes)...")
            sftp.put(str(path), posixpath.join(REMOTE_JOB_DIR, path.name))
    finally:
        sftp.close()
    return [p.name for p in files]


def upload_pseudos(ssh, pseudo_dir, log):
    """Upload *.UPF files from pseudo_dir to ~/qe_job/pseudo/ so the
    pseudo_dir='./pseudo/' setting in the .in files resolves on the VM."""
    files = sorted(Path(pseudo_dir).glob("*.UPF"))
    if not files:
        raise RuntimeError(
            f"No .UPF files in {pseudo_dir} — QE runs need pseudopotentials "
            f"(see qe/pseudopotentials/SOURCES.md for where to get them)"
        )
    remote_pseudo = posixpath.join(REMOTE_JOB_DIR, "pseudo")
    run_quiet(ssh, f"mkdir -p {remote_pseudo}")
    sftp = ssh.open_sftp()
    try:
        for path in files:
            log(f"Uploading pseudo/{path.name} ({path.stat().st_size} bytes)...")
            sftp.put(str(path), posixpath.join(remote_pseudo, path.name))
    finally:
        sftp.close()
    return [p.name for p in files]


def _sftp_write(ssh, remote_path, content):
    sftp = ssh.open_sftp()
    try:
        with sftp.open(remote_path, "w") as fh:
            fh.write(content)
    finally:
        sftp.close()


def _find_attachable(ssh, base, command):
    """If a prior launch of this exact command exists on the VM, return
    'finished' or 'running'; else None. Lets a rerun after a local crash
    pick up the remote job instead of starting over."""
    _, prev_cmd = run_quiet(ssh, f"cat {REMOTE_JOB_DIR}/{base}.cmd 2>/dev/null")
    if prev_cmd.strip() != command.strip():
        return None
    status, _ = run_quiet(ssh, f"test -f {REMOTE_JOB_DIR}/{base}.exit")
    if status == 0:
        return "finished"
    _, pid = run_quiet(ssh, f"cat {REMOTE_JOB_DIR}/{base}.pid 2>/dev/null")
    if pid.strip().isdigit():
        alive, _ = run_quiet(ssh, f"kill -0 {pid.strip()} 2>/dev/null")
        if alive == 0:
            return "running"
    return None


def run_remote_command(ssh, command, log, output_name):
    """Run one command in ~/qe_job DETACHED (setsid+nohup wrapper script)
    and poll for its exit status, streaming new output as it appears.

    A local run_job.py death can NOT kill the remote job: the command's
    output accumulates in <output_name> on the VM and its exit status is
    written to <base>.exit by the wrapper. On a rerun with the same
    command, a still-running instance is attached to (not relaunched) and
    a finished one has its recorded exit status returned. (To force a
    fresh rerun of an identical command, delete ~/qe_job/<base>.exit and
    <base>.cmd on the VM first.)

    Returns the exit status."""
    base = output_name[:-4] if output_name.endswith(".log") else output_name
    remote_base = f"{REMOTE_JOB_DIR}/{base}"

    state = _find_attachable(ssh, base, command)
    if state:
        log(f"Attaching to {state} instance of {command!r} on the VM "
            f"(not relaunching).")
    else:
        script = (
            "#!/bin/bash\n"
            f'cd "$HOME/{REMOTE_JOB_DIR}"\n'
            f"echo $$ > {base}.pid\n"
            "set -o pipefail\n"
            f"({command}) >> {output_name} 2>&1\n"
            f"echo $? > {base}.exit\n"
        )
        _sftp_write(ssh, f"{remote_base}.sh", script)
        _sftp_write(ssh, f"{remote_base}.cmd", command)
        run_quiet(
            ssh,
            f"cd {REMOTE_JOB_DIR} && rm -f {base}.exit {base}.pid && "
            f": > {output_name} && "
            f"setsid nohup bash {base}.sh </dev/null >/dev/null 2>&1 &",
        )
        log(f"Launched detached: {command}   (output -> {output_name}; "
            f"survives local disconnects)")

    offset = 0
    while True:
        try:
            # stream any new output bytes since the last poll
            _, size_txt = run_quiet(
                ssh, f"wc -c < {REMOTE_JOB_DIR}/{output_name} 2>/dev/null")
            size = int(size_txt.strip() or 0)
            if size > offset:
                _, stdout, _ = ssh.exec_command(
                    f"tail -c +{offset + 1} {REMOTE_JOB_DIR}/{output_name}")
                sys.stdout.write(stdout.read().decode(errors="replace"))
                sys.stdout.flush()
                offset = size
            done, status_txt = run_quiet(
                ssh, f"cat {remote_base}.exit 2>/dev/null")
            if done == 0 and status_txt.strip():
                return int(status_txt.strip())
        except Exception as exc:
            # transient SSH drop — SshSession reconnects on the next call
            log(f"Polling hiccup ({exc!r}); retrying in {REMOTE_POLL_S}s...")
        time.sleep(REMOTE_POLL_S)


def run_command_sequence(ssh, commands, log):
    """Run commands in order on the same VM; stop at the first failure.
    Returns [(command, exit_status), ...] for the commands that ran."""
    statuses = []
    total = len(commands)
    for i, command in enumerate(commands, 1):
        status = run_remote_command(ssh, command, log, f"run_stdout_{i}.log")
        statuses.append((command, status))
        log(f"Command {i}/{total} exit status {status}: {command}")
        if status != 0:
            skipped = commands[i:]
            if skipped:
                log(f"Skipping {len(skipped)} remaining command(s) after failure: "
                    + "; ".join(skipped))
            break
    return statuses


BOOKKEEPING_SUFFIXES = (".sh", ".pid", ".exit", ".cmd")


def download_outputs(ssh, input_dir, uploaded, log):
    """Download new top-level files from ~/qe_job back into input_dir, so
    outputs (scf.out, run_stdout_*.log, ...) sit next to their inputs.
    The detached-execution bookkeeping files (run_stdout_*.sh/.pid/.exit/
    .cmd) are implementation detail, not results — skipped."""
    input_dir = Path(input_dir)
    sftp = ssh.open_sftp()
    downloaded, skipped_dirs = [], []
    try:
        for entry in sftp.listdir_attr(REMOTE_JOB_DIR):
            if stat.S_ISDIR(entry.st_mode):
                skipped_dirs.append(entry.filename)
                continue
            if entry.filename in uploaded:
                continue
            if (entry.filename.startswith("run_stdout_")
                    and entry.filename.endswith(BOOKKEEPING_SUFFIXES)):
                continue
            log(f"Downloading {entry.filename} ({entry.st_size} bytes)...")
            sftp.get(
                posixpath.join(REMOTE_JOB_DIR, entry.filename),
                str(input_dir / entry.filename),
            )
            downloaded.append(entry.filename)
    finally:
        sftp.close()
    if skipped_dirs:
        log(
            "Skipped remote directories (bulk QE artifacts, gitignored anyway): "
            + ", ".join(sorted(skipped_dirs))
        )
    return downloaded


def detect_qe_banner(input_dir):
    """Pull the 'Program PWSCF/PHONON ... starts' banners from the downloaded
    per-command output logs."""
    banners = []
    for path in sorted(Path(input_dir).glob("run_stdout_*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "Program" in line and "starts" in line:
                banners.append(line.strip())
                break
    return " | ".join(banners) if banners else None


def main():
    parser = argparse.ArgumentParser(
        description="Run QE command(s) on a throwaway Azure VM (teardown guaranteed)."
    )
    parser.add_argument("--input-dir", required=True,
                        help="local directory with ALL QE inputs (scf.in, ph.in, ...), "
                             "e.g. qe/SrTiO3/harmonic_gamma — uploaded once, up front")
    parser.add_argument("--command", action="append", required=True,
                        help='remote command; repeat for a chain run in order on the '
                             'same VM, e.g. --command "mpirun -np 4 pw.x -i scf.in" '
                             '--command "mpirun -np 4 ph.x -i ph.in" '
                             '(-np = physical cores = vCPUs/2 on Fsv2)')
    parser.add_argument("--pseudo-dir", default="qe/pseudopotentials",
                        help="local dir whose *.UPF files are uploaded to "
                             "<remote job dir>/pseudo/ on every run "
                             "(default: %(default)s)")
    parser.add_argument("--on-failure", choices=("keep", "teardown"), default="keep",
                        help="what to do with the VM when a COMMAND exits non-zero "
                             "(default: %(default)s — the VM stays up and BILLING "
                             "for diagnosis; infra errors always tear down)")
    parser.add_argument("--vm-size", default=DEFAULT_VM_SIZE)
    parser.add_argument("--vm-name",
                        help="override the generated VM name (also used to "
                             "derive NIC/IP/disk names)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"--input-dir {input_dir} is not a directory")
    if not Path(args.pseudo_dir).is_dir():
        parser.error(f"--pseudo-dir {args.pseudo_dir} is not a directory")

    cfg = load_config()  # fail fast on missing .env before touching Azure
    vm_name = args.vm_name or (
        f"qe-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        f"-{secrets.token_hex(2)}"
    )
    names = resource_names(vm_name)
    log = RunLog(input_dir / "run_log.txt").log

    log("=" * 60)
    log(f"Job start: input={input_dir}")
    for i, command in enumerate(args.command, 1):
        log(f"  command {i}/{len(args.command)}: {command}")
    log(f"VM name: {vm_name}  size: {args.vm_size}  region: {cfg['LOCATION']}  "
        f"group: {cfg['AZURE_GROUP']}")

    provisioned = False
    vm_ip = None
    failed_command = None  # (index, command, exit_status) of first failure
    exit_code = 0
    try:
        # Set BEFORE provisioning: a partial failure (e.g. IP created, NIC
        # attach failed) must still trigger cleanup. teardown_vm checks each
        # resource by name, so tearing down nothing is harmless.
        provisioned = True
        vm_ip = provision_vm(vm_name, vm_size=args.vm_size, cfg=cfg, log=log)
        qe_version = wait_for_qe_ready(vm_ip, log)
        # Lets the user ssh in and tail run_stdout_*.log while the job runs.
        print_connection_info(vm_name, args.vm_size, cfg["LOCATION"], vm_ip, log=log)

        ssh = SshSession(vm_ip)  # auto-reconnects if the connection drops
        try:
            check_mpi_slots(ssh, args.command, log)
            uploaded = upload_inputs(ssh, input_dir, log)
            upload_pseudos(ssh, args.pseudo_dir, log)
            statuses = run_command_sequence(ssh, args.command, log)
            # Record before downloading: even if the download then fails,
            # the finally block must know a command failure happened so
            # --on-failure keep can preserve the VM (outputs still on it).
            if statuses and statuses[-1][1] != 0:
                failed_command = (len(statuses), *statuses[-1])
            downloaded = download_outputs(ssh, input_dir, uploaded, log)
        finally:
            ssh.close()

        banner = detect_qe_banner(input_dir)
        log(f"QE version: quantum-espresso {qe_version}"
            + (f" | {banner}" if banner else ""))
        log(f"Outputs in {input_dir}: {', '.join(downloaded) or '(none)'}")
        failed = [(c, s) for c, s in statuses if s != 0]
        if failed:
            raise RuntimeError(
                f"{len(failed)} command(s) failed: "
                + "; ".join(f"{c!r} (exit {s})" for c, s in failed)
            )
        log(f"Job SUCCEEDED ({len(statuses)}/{len(args.command)} commands, all exit 0).")
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C) — proceeding to teardown.")
        exit_code = 130
    except Exception as exc:
        log(f"Job FAILED: {exc!r}")
        exit_code = 1
    finally:
        keep_vm = provisioned and failed_command and args.on_failure == "keep"
        if not provisioned:
            log("Nothing to tear down.")
        elif keep_vm:
            idx, command, status = failed_command
            log(f"on-failure=keep: VM {vm_name} LEFT RUNNING for diagnosis — "
                f"command {idx} ({command!r}) exited {status}.")
            print_connection_info(vm_name, args.vm_size, cfg["LOCATION"],
                                  vm_ip, log=log)
            failed_log = input_dir / f"run_stdout_{idx}.log"
            if failed_log.is_file():
                tail = failed_log.read_text(errors="replace").splitlines()[-40:]
                print(f"--- last {len(tail)} lines of {failed_log.name} ---")
                print("\n".join(tail))
                print("--- end of log tail ---")
            else:
                log(f"({failed_log.name} was not downloaded — read it on the "
                    f"VM at qe_job/run_stdout_{idx}.log)")
            log(f"VM is running and BILLING; ssh in to diagnose, or run "
                f"`python azure/teardown_vm.py {vm_name}` when done.")
        else:
            if failed_command:
                log("on-failure=teardown: cleaning up despite the command failure.")
            try:
                deleted = teardown_vm(vm_name, cfg=cfg, log=log)
                log("Teardown complete. Deleted resource IDs:")
                for rid in deleted:
                    log(f"  {rid}")
            except Exception as exc:
                exit_code = exit_code or 2
                log(f"TEARDOWN FAILED: {exc}")
                log(f"MANUALLY delete these in group {cfg['AZURE_GROUP']}: "
                    f"{names['vm']}, {names['nic']}, {names['public_ip']}, "
                    f"{names['os_disk']}  "
                    f"(retry: .venv/bin/python azure/teardown_vm.py {vm_name})")
        log(f"Job end (exit code {exit_code}).")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
