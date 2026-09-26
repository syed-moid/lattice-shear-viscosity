"""Shared paths for the diagnostic scripts: run from the repository root as
`uv run python scripts/v3_diagnostics/<name>.py`; outputs go to data/processed/v3_diagnostics/."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIAG = REPO / "data" / "processed" / "v3_diagnostics"
SCPH = REPO / "scripts" / "scph"
RAW = REPO / "data" / "raw" / "alamode_sto"
for p in (HERE, SCPH, REPO / "scripts", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class _Tee:
    def __init__(self, path):
        self._f = open(path, "w")
        self._o = sys.stdout

    def write(self, s):
        self._o.write(s)
        self._f.write(s)

    def flush(self):
        self._o.flush()
        self._f.flush()


def tee_stdout(name):
    """copy everything printed by the script to data/processed/v3_diagnostics/<name>."""
    DIAG.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(DIAG / name)
