#!/usr/bin/env python3
"""Dispersion of the hybrid model (production surface of compute_eta_SrTiO3.py) along Gamma-X-M-Gamma-R-X:
bare harmonic frequencies of the example harmonic set (re-expressed on the 4x4x4 supercell) and the SCPH-renormalised
frequencies at 300 K (correction mesh 2^3, inner mesh 12^3, SELF_OFFDIAG = 1), on the path of
data/processed/harmonic_dispersion_SrTiO3.csv. Negative values denote imaginary (unstable) harmonic modes.
Writes data/processed/dispersion_hybrid_SrTiO3.csv.
Usage: uv run python scripts/export_hybrid_dispersion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    path = pd.read_csv(REPO / "data" / "processed" / "harmonic_dispersion_SrTiO3.csv", comment="#")
    pts = path.drop_duplicates("path_index")[["path_index", "qx", "qy", "qz", "path_coord"]].sort_values("path_index")
    bare = eta_mod.qe_set("reference")
    ren = AlamodeSet(eta_mod.own_od1_xml("reference", 300), eta_mod.fc_paths()["reference"])
    rows = []
    qarr = pts[["qx", "qy", "qz"]].values.astype(float)
    for k, (_, r) in enumerate(pts.iterrows()):
        q = (float(r.qx), float(r.qy), float(r.qz))       # 2 pi/a units = reduced units of the cubic cell
        if np.linalg.norm(q) < 1e-9:                       # Gamma: approach along the path (non-analytic term)
            nb = qarr[k + 1] if k + 1 < len(qarr) and np.linalg.norm(qarr[k + 1]) > 1e-9 else qarr[k - 1]
            q = tuple(1e-3 * nb / np.linalg.norm(nb))
        wb = np.sort(diagonalise(bare.dynmat(q))[0])
        wr = np.sort(diagonalise(ren.dynmat(q, asr_onsite=True))[0])
        for b in range(15):
            rows.append({"path_index": int(r.path_index), "qx": q[0], "qy": q[1], "qz": q[2], "path_coord": float(r.path_coord),
                         "branch": b + 1, "omega_bare_cm1": wb[b], "omega_scph300_cm1": wr[b]})
    out = REPO / "data" / "processed" / "dispersion_hybrid_SrTiO3.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# dispersion_hybrid_SrTiO3.csv - produced by scripts/export_hybrid_dispersion.py\n"
                 "# hybrid model: example harmonic set on 4x4x4 (bare) and its SCPH renormalisation at 300 K (correction mesh 2^3,\n"
                 "# inner mesh 12^3, SELF_OFFDIAG = 1); q in 2 pi/a; omega < 0 denotes an imaginary harmonic mode.\n")
        pd.DataFrame(rows).to_csv(fh, index=False, float_format="%.4f")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
