#!/usr/bin/env python3
"""Table 2: q-mesh convergence of eta_xyxy(SrTiO3, 300 K) on the production model (hybrid model tut_z_od1,
SELF_OFFDIAG = 1, construction C) + isotope-smearing sensitivity.

Mesh sequence: Gamma-centred 9^3, 11^3 (production), 13^3, 15^3. The dynamical matrices and couplings
are evaluated by the builder at every q (no per-mesh QE files are needed), with the identical assembly
(same surface, same linewidth map, same Vogt exception). Smearing: the Gaussian width of the delta in
the eigenvector-resolved isotope rate (production 10 cm-1) rerun at 5 and 20 cm-1, reported as the
change of the f = 0.15 suppression.
Writes: data/processed/table1_convergence_SrTiO3.csv (file name kept; manuscript Table 2)
Usage: uv run python scripts/table1_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
import compute_eta_isotope_SrTiO3 as iso_mod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
MESHES = [9, 11, 13, 15]


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    surface = eta_mod.load_construction_surface(T_K)
    rows_out = ["mesh,n_q,eta_300K_Pas,dev_from_densest_pct"]
    results = []
    modes11 = None
    for n in MESHES:
        modes = eta_mod.construction_modes(T_K, n, surface)
        if n == 11:
            modes11 = modes
        eta, _, _ = eta_mod.assemble_construction(T_K, mesh_n=n, modes=modes, surface=surface, vogt=vogt)
        results.append((n, eta))
        print(f"{n:2d}^3: n_q = {n ** 3:5d}  eta_C(300 K) = {eta:.4e} Pa s")
    eta_densest = results[-1][1]
    print(f"\ndeviation from the densest mesh (15^3):")
    for n, eta in results:
        dev = 100.0 * (eta / eta_densest - 1.0)
        print(f"  {n:2d}^3: {dev:+6.2f} %")
        rows_out.append(f"{n}^3,{n ** 3},{eta:.6e},{dev:.3f}")
    print("\nisotope-channel smearing sensitivity (f = 0.15, 11^3, production-surface eigenvectors):")
    sig_rows = ["sigma_cm1,eta_f015_Pas,suppression_pct"]
    eta0, _, _ = eta_mod.assemble_construction(T_K, modes=modes11, surface=surface, vogt=vogt)
    for sigma in [5.0, 10.0, 20.0]:
        ratio = iso_mod.isotope_series("renormalised", fractions=[0.15], sigma_cm1=sigma, modes=modes11,
                                       surface=surface, vogt=vogt)[0][0.15]
        sup = 100.0 * (ratio - 1.0)
        print(f"  sigma = {sigma:4.0f} cm-1: eta(f=0.15) = {eta0 * ratio:.4e}  suppression = {sup:+.2f}%{' (production)' if sigma == 10.0 else ''}")
        sig_rows.append(f"{sigma},{eta0 * ratio:.6e},{sup:.3f}")
    out = REPO / "data" / "processed" / "table1_convergence_SrTiO3.csv"
    header = [
        "# table1_convergence_SrTiO3.csv - produced by scripts/table1_convergence.py (manuscript Table 2)",
        "# q-mesh convergence of eta_xyxy(300 K), production model: hybrid model tut_z_od1 (SELF_OFFDIAG = 1, inner mesh 12^3),",
        "# construction C, h = 0.010 engineering shear; identical assembly at every mesh (builder-evaluated",
        "# dynamical matrices and couplings). Smearing block: isotope-channel Gaussian delta-width",
        "# sensitivity at f = 0.15 (projected rate, production-surface eigenvectors).",
    ]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(rows_out) + "\n\n" + "\n".join(sig_rows) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
