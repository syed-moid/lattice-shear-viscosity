#!/usr/bin/env python3
"""Exact-degeneracy threshold check for the coupling rule of constructions B and C.

For every renormalised reference set used in the paper (hybrid model: inner meshes 2^3-12^3 at 300 K and the 2^3/12^3
temperature series; diagnostic surface: inner meshes 2^3-12^3 at 300 K), the renormalised frequencies on the 11^3 mesh
are sorted per wave vector and every gap between neighbouring frequencies is recorded. The exact-degeneracy threshold
1e-3 cm^-1 is valid for a set when no gap falls in the window (1e-4, 1e-2) cm^-1 (SCPH round-off below, genuine
splittings above).

Output: data/processed/v3_diagnostics/degeneracy_gap_check.csv
Usage: uv run python scripts/v3_diagnostics/degeneracy_gap_check.py
Needs: data/raw/alamode_sto (raw archive).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG, RAW  # noqa: E402  (sets sys.path)

import compute_eta_SrTiO3 as E  # noqa: E402
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

Z, OWN = RAW / "z_tut", RAW / "own_od1"


def sets():
    for mesh in ("i2s2", "i2s4", "i2s8", "i2s12"):
        yield "hybrid", mesh, 300, Z / mesh / f"renorm_z_reference_{mesh}_od1_300K.xml", E.Z_FC["reference"]
    for T in (250, 350, 400):
        yield "hybrid", "i2s12", T, Z / "i2s12_T" / f"renorm_z_reference_i2s12T{T}_od1_{T}K.xml", E.Z_FC["reference"]
    for mesh in ("i2s2", "i2s4", "i2s8", "i2s12"):
        yield "diagnostic", mesh, 300, OWN / mesh / f"renorm_reference_{mesh}_od1_300K.xml", E.QE_FC["reference"]


rows = []
for surf, mesh, T, xml, fc in sets():
    s = AlamodeSet(xml, fc)
    gaps = []
    for q in E.mesh_points(11):
        w = np.sort(diagonalise(s.dynmat(q, asr_onsite=True))[0])
        gaps.append(np.diff(w))
    g = np.concatenate(gaps)
    rows.append({"surface": surf, "mesh": mesh, "T_K": T, "n_gaps": len(g),
                 "n_below_1e-4": int((g <= 1e-4).sum()), "n_in_window_1e-4_1e-2": int(((g > 1e-4) & (g < 1e-2)).sum()),
                 "n_1e-2_to_0.5": int(((g >= 1e-2) & (g < 0.5)).sum()), "max_gap_below_1e-4": float(g[g <= 1e-4].max()),
                 "min_gap_above_1e-2": float(g[g >= 1e-2].min())})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
out = DIAG / "degeneracy_gap_check.csv"
tmp = out.with_suffix(".tmp")
with open(tmp, "w") as fh:
    fh.write("# degeneracy_gap_check.py: gaps between neighbouring renormalised frequencies (cm^-1) on the 11^3 mesh per set;\n"
             "# the exact-degeneracy threshold 1e-3 cm^-1 is valid where n_in_window_1e-4_1e-2 = 0.\n")
    df.to_csv(fh, index=False, float_format="%.4g")
tmp.rename(out)
print(df.to_string(index=False))
