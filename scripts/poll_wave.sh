#!/bin/bash
# scripts/poll_wave.sh — poll one or more Azure QE job VMs by name and
# self-log to monitoring/status_log.md (no external redirection needed).
#
# Usage: bash scripts/poll_wave.sh <vm-name> [<vm-name> ...]
#
# Each VM is polled with `poll_job.py <vm> 4` — every A5' strain job runs
# exactly 4 commands (scf, forces check, ph.x dispersion, q2r). Prints one
# status block per VM to stdout (poll_job.py exit code: 0 = all done,
# 1 = still running, 2 = a command failed) AND appends one line per VM to
# monitoring/status_log.md, timestamped and stamped with each command's
# state, so the log stays current without a separate write step.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${REPO_ROOT}/monitoring/status_log.md"
TS="$(date -u '+%Y-%m-%d %H:%M UTC')"

for vm in "$@"; do
    echo "== ${vm} =="
    status_lines="$(uv run --group azure python "${REPO_ROOT}/dft/azure/poll_job.py" "${vm}" 4)"
    exit_code=$?
    echo "${status_lines}"
    echo "exit: ${exit_code}"
    echo

    summary="$(echo "${status_lines}" | tr '\n' ' ' | sed 's/  */ /g')"
    case "${exit_code}" in
        0) label="DONE (all commands exit 0)" ;;
        2) label="FAILED (a command exited nonzero)" ;;
        *) label="running" ;;
    esac
    printf '| %s | %s | poll | %s — %s |\n' "${TS}" "${vm}" "${label}" "${summary}" >> "${LOG}"
done
