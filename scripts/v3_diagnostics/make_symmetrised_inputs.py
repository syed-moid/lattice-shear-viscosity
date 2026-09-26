#!/usr/bin/env python3
"""make_symmetrised_inputs.py - strained harmonic inputs with the even offset removed.

On the q2r 4x4x4 grid (identical storage for all sets) the short-range force constants, the dielectric
tensor and the Born charges of the +-0.005 sets are replaced by
    X_+- = X_0 +- (X_+ - X_-) / 2
(the reference set plus/minus the odd part), keeping each strained set's own lattice and positions.
The converted sets are then produced exactly as the production sets (dipole restored on the grid,
distributed minimal-norm sum rule) as shear_xy_{p,m}005_sym_full_fc2.xml with BORN files.
Writes into DYNMAT_SCRATCH/run/convert.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from coupling_kernel import FC  # noqa: E402
from q2r_to_alamode_fc2 import add_dipole_real_space, read_qe_fc, write_alamode_xml, write_born  # noqa: E402

OUT = Path(os.environ["DYNMAT_SCRATCH"]) / "run" / "convert"


def main():
    d0, dp, dm = (read_qe_fc(FC[t]) for t in ("reference", "shear_xy_p005", "shear_xy_m005"))
    for sign, base, name in ((+1, dp, "shear_xy_p005_sym"), (-1, dm, "shear_xy_m005_sym")):
        d = dict(base)
        d["fc"] = d0["fc"] + sign * 0.5 * (dp["fc"] - dm["fc"])
        d["dielectric"] = d0["dielectric"] + sign * 0.5 * (dp["dielectric"] - dm["dielectric"])
        d["born"] = d0["born"] + sign * 0.5 * (dp["born"] - dm["born"])
        even_fc = np.abs(base["fc"] - d["fc"]).max()
        d = add_dipole_real_space(d)
        n = write_alamode_xml(d, OUT / f"{name}_full_fc2.xml", asr="distributed", source=f"{name} (symmetrised)")
        write_born(d, OUT / f"BORN_{name}")
        print(f"{name}: max |even part removed| = {even_fc:.3e} Ry/bohr^2; {n} FC2 entries")


if __name__ == "__main__":
    main()
