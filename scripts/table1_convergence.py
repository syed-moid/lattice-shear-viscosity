#!/usr/bin/env python3
"""Table 1: q-mesh convergence of eta_xyxy(SrTiO3, 300 K) + isotope-smearing
sensitivity — fills the [DATA] slot in manuscript section 3.4.

Mesh sequence: 4^3 (exactly DFPT-commensurate — zero Fourier-interpolation
error, so it differs from the interpolated meshes for two reasons at once
and is reported as a separate line, not part of the Richardson-style
convergence read-off), then interpolated Gamma-centered 7^3, 9^3, 11^3
(production), 13^3. All meshes use the identical generation-3 assembly
(scripts/compute_eta_SrTiO3.py) — same maps, same partition, same
Vogt/theory-floor conventions; only the q-sampling changes.

Smearing sensitivity: the isotope channel's Gaussian DOS smearing
(production sigma = 10 cm-1) rerun at 5 and 20 cm-1, reported as the
change in the f = 0.15 suppression.

Reads : data/raw/gruneisen_modes/SrTiO3/{,ongrid4,mesh7,mesh9,mesh13}
        + the ALAMODE per-T files (via compute_eta_SrTiO3)
Writes: data/processed/table1_convergence_SrTiO3.csv

Usage: uv run python scripts/table1_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
import compute_eta_isotope_SrTiO3 as iso_mod  # noqa: E402
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300

MESHES = [
    ("4^3 (exact DFPT grid)", "ongrid4", 4),
    ("7^3", "mesh7", 7),
    ("9^3", "mesh9", 9),
    ("11^3 (production)", "", 11),
    ("13^3", "mesh13", 13),
]


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    rows_out = ["mesh,n_q,eta_300K_Pas,dev_from_133_pct"]
    results = []
    for label, sub, n in MESHES:
        directory = MODES_DIR / "SrTiO3" / sub if sub else MODES_DIR / "SrTiO3"
        if not (directory / "reference.modes").exists():
            print(f"{label}: MISSING ({directory})")
            continue
        rows = compute_dataset(directory, MASSES["SrTiO3"], mesh_n=n)
        eta_mod.N_Q = n**3
        eta, sec, flags = eta_mod.assemble(T_K, rows, vogt)
        results.append((label, n**3, eta))
        print(f"{label:>22}: n_q = {n**3:5d}  eta(300K) = {eta:.4e} Pa s")
    eta_mod.N_Q = 11**3

    eta_densest = results[-1][2]
    print(f"\ndeviation from the densest mesh ({results[-1][0]}):")
    for label, nq, eta in results:
        dev = 100.0 * (eta / eta_densest - 1.0)
        print(f"  {label:>22}: {dev:+6.2f}%")
        rows_out.append(f"\"{label}\",{nq},{eta:.6e},{dev:.3f}")

    # ---- isotope-smearing sensitivity (production 11^3 mesh) ----
    print("\nisotope-channel smearing sensitivity (f = 0.15, 11^3):")
    from latvisc.isotope import isotope_scattering_rate, mass_variance_g2
    import numpy as np
    rows11 = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    g2 = mass_variance_g2([iso_mod.M_NAT_O, iso_mod.M_18O], [0.85, 0.15])
    sig_rows = ["sigma_cm1,eta_f015_Pas,suppression_pct"]
    eta0, _, _ = eta_mod.assemble(T_K, rows11, vogt)
    for sigma in [5.0, 10.0, 20.0]:
        iso_mod.DOS_SIGMA_CM1 = sigma
        dos = iso_mod.build_dos(rows11)

        def extra(omega_r_cm1, g2=g2, dos=dos):
            w = omega_r_cm1 * eta_mod.CM1
            rate = isotope_scattering_rate(w, g2, iso_mod.V0, dos(w))
            return float(np.squeeze(rate)) / 2.0 / eta_mod.CM1

        eta_f, _, _ = eta_mod.assemble(T_K, rows11, vogt, extra_gamma_hwhm_cm1=extra)
        sup = 100.0 * (eta_f / eta0 - 1.0)
        tag = " (production)" if sigma == 10.0 else ""
        print(f"  sigma = {sigma:4.0f} cm-1: eta(f=0.15) = {eta_f:.4e}  "
              f"suppression = {sup:+.2f}%{tag}")
        sig_rows.append(f"{sigma},{eta_f:.6e},{sup:.3f}")
    iso_mod.DOS_SIGMA_CM1 = 10.0

    out = REPO / "data" / "processed" / "table1_convergence_SrTiO3.csv"
    header = [
        "# table1_convergence_SrTiO3.csv - produced by scripts/table1_convergence.py",
        "# q-mesh convergence of eta_xyxy(300 K), generation-3 assembly; the 4^3 line",
        "# is the exactly-DFPT-commensurate grid (zero interpolation error) and is",
        "# NOT part of the interpolated-mesh convergence read-off. Smearing block:",
        "# isotope-channel Gaussian DOS smearing sensitivity at f = 0.15.",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(rows_out) + "\n\n"
                   + "\n".join(sig_rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
