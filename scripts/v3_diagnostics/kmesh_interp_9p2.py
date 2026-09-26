#!/usr/bin/env python3
"""kmesh_interp_9p2.py - interpolation-mesh sensitivity at fixed inner mesh, read from kmesh_convergence_9p1.csv
(rows with KMESH_SCPH = 12 and 16 across KMESH_INTERPOLATE = 2, 3, 4), plus the memory feasibility of the variants
not run. Writes data/processed/v3_diagnostics/kmesh_interp_9p2.csv."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
df = pd.read_csv(DIAG / "kmesh_convergence_9p1.csv", comment="#")
cols = ["mesh", "KMESH_INTERPOLATE", "KMESH_SCPH", "eta_C_a_Pas", "eta_C_b_Pas", "eta_B_a_Pas", "eta_A_a_Pas", "R_AFD_cm1", "TO1_cm1",
        "X_lowest_cm1", "M_lowest_cm1", "kappa_300K_W_mK", "n_imag_solution_grid", "n_imag_8cubed", "n_imag_11cubed", "iter_ref", "resid_ref", "wall_s_ref"]
sub = df[df.KMESH_SCPH.isin([12, 16])][cols].copy()
sub["status"] = "run"
NIRR = {2: 4, 3: 4, 4: 10, 6: 20, 8: 35, 12: 84, 16: 172}
rows = []
for kint, ks in [(2, 12), (3, 12), (4, 12), (6, 12), (12, 12), (2, 16), (4, 16), (8, 16), (16, 16)]:
    gb = NIRR[kint] * ks ** 3 * 15 ** 4 * 16 / 1e9
    tag = f"i{kint}s{ks}"
    if tag in set(sub.mesh):
        sub.loc[sub.mesh == tag, "peak_GB_per_rank"] = 2 * gb
        continue
    rows.append({"mesh": tag, "KMESH_INTERPOLATE": kint, "KMESH_SCPH": ks, "status": f"not run: {2 * gb:.0f} GB peak per rank ({gb:.0f} GB persistent) vs 36 GB RAM" if 2 * gb > 30 else "not run (not requested)", "peak_GB_per_rank": 2 * gb})
out = pd.concat([sub, pd.DataFrame(rows)], ignore_index=True).sort_values(["KMESH_SCPH", "KMESH_INTERPOLATE"])
o = DIAG / "kmesh_interp_9p2.csv"; tmp = o.with_suffix(".tmp")
with open(tmp, "w") as fh:
    fh.write("# (kmesh_interp_9p2.py): interpolation-mesh sensitivity at fixed inner mesh (12^3, 16^3), own surface, SELF_OFFDIAG = 1, 300 K;\n"
             "# rows copied from kmesh_convergence_9p1.csv; peak memory = 2 x n_irred(KMESH_INTERPOLATE) x KMESH_SCPH^3 x 15^4 x 16 B per MPI rank\n"
             "# (anphon v4_array_all + v4_mpi). ALAMODE 1.5.0 documentation (docs/source/anphondir/inputanphon.rst, KMESH_SCPH): 'This k mesh is used\n"
             "# for the inner loop of the SCPH equation. Each value of KMESH_SCPH must be equal to or a multiple of the number of KMESH_INTERPOLATE\n"
             "# in the same direction.' n_irred of the 6x6x6/12x12x12/16x16x16 grids (20/84/172) estimated from the cubic point group.\n")
    out.to_csv(fh, index=False, float_format="%.6g")
tmp.rename(o); print(out.to_string())
