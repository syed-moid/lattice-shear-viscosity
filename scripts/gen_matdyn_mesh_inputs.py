#!/usr/bin/env python3
"""Stage matdyn.x mesh inputs from strained/reference .444.fc files.

Single source of truth for q-point conventions is
src/latvisc/matdyn_input.py — this script is the ONLY place that turns a
.444.fc file into a matdyn.in, so the free nonlinearity diagnostic and the
eventual full-mesh Grueneisen production run can never diverge in how they
sample q (see the G1' incident log in data/processed/reports/GATE_1p.md
for why that matters: Cartesian band-form q silently breaks cross-strain
mode continuity under shear strain).

Usage:
  uv run python scripts/gen_matdyn_mesh_inputs.py <material> [--mesh N] \
      <label>=<path-to-.444.fc> [<label>=<path> ...]

--mesh N (default 11): Gamma-centered N x N x N fractional mesh. Pass
--mesh 4 to sample exactly the original 4x4x4 DFPT coarse grid (every
q-point commensurate with the real-space force constants, so matdyn
returns the exact dynamical matrix with no Fourier-interpolation error —
useful for separating genuine anharmonic curvature from fc-interpolation
noise, see the G1' incident log in data/processed/reports/GATE_1p.md).

Writes, per label, into data/raw/gruneisen_modes/<material>/staging/
(or staging_mesh<N>/ when --mesh is not 11):
  <material>_<label>.444.fc   (copy of the input force-constant file)
  matdyn_<label>.in           (flvec = <label>.modes, 11x11x11 Gamma-centered
                                fractional mesh)

The staging directory is ready to upload as a single Azure mini-job
(matdyn.x is fast — order of a minute for the full mesh, negligible cost).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.matdyn_input import gamma_centered_mesh, write_matdyn_input  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    material = sys.argv[1]
    args = sys.argv[2:]
    mesh_n = 11
    if args and args[0] == "--mesh":
        mesh_n = int(args[1])
        args = args[2:]
    pairs = dict(arg.split("=", 1) for arg in args)

    staging_name = "staging" if mesh_n == 11 else f"staging_mesh{mesh_n}"
    staging = REPO / "data" / "raw" / "gruneisen_modes" / material / staging_name
    staging.mkdir(parents=True, exist_ok=True)
    qpoints = gamma_centered_mesh(mesh_n)

    for label, fc_path in pairs.items():
        fc_name = f"{material}_{label}.444.fc"
        shutil.copy(Path(fc_path), staging / fc_name)
        write_matdyn_input(
            staging / f"matdyn_{label}.in",
            flfrc=fc_name,
            flfrq=f"{material}_{label}.mesh11.freq",
            flvec=f"{label}.modes",
            qpoints=qpoints,
        )
    print(f"Staged {len(pairs)} matdyn input(s) ({len(qpoints)} q-points each) "
          f"in {staging.relative_to(REPO)}")


if __name__ == "__main__":
    main()
