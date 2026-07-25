#!/usr/bin/env python3
"""Rebuild data/processed/softmode_inputs_BaTiO3.csv — the COMPLETE
zone-center soft-mode input set for the BaTiO3 Route-H assembly.

Sources (all cubic phase; per-row provenance):
  * zone-center omega_s(T): the canonical merged digitizations
    (data/processed/bto_softmode_digitized/, from VSR PRB 26, 5904 (1982)
    Fig. 3 + Presting PRB 28, 6097 (1983) Fig. 2 — 4/4 built-in
    consistency checks passed, see merge_bto_softmode_digitizations.py);
  * zone-center Gamma_s(T): rel_damping x omega_s / 2 (VSR Fig. 3 series)
    and absolute gamma / 2 (Presting Fig. 2 series);
  * the finite-q Harada 1971 neutron anchors and the Ehsan 2021 SCP
    anchors retained from the 2026-07-23 import (labeled; NOT zone-center
    omega_s(T) inputs — context anchors).

Conventions applied HERE, in code (raw and canonical files untouched):
  * gamma quantities are classical-dispersion-oscillator FULL damping
    constants -> Gamma_HWHM = gamma_full/2 (pipeline convention,
    tau = 1/(2 Gamma_HWHM));
  * overdamped flag per instruction: gamma_full/Omega0 > 1 (every VSR
    zone-center point trips it; the tau_eff of Eq. (12) handles the
    regimes continuously regardless);
  * temperatures already Kelvin (verified against the figure images —
    see references/digitized/*/README.md).

Usage: uv run python scripts/build_bto_softmode_inputs.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIG = REPO / "data" / "processed" / "bto_softmode_digitized"
OUT = REPO / "data" / "processed" / "softmode_inputs_BaTiO3.csv"
MEV_TO_CM1 = 8.065544


def read_canonical(name):
    rows = []
    for line in (DIG / f"{name}.csv").read_text().splitlines():
        if line.startswith("#") or line.startswith("T_K"):
            continue
        t, v, src = line.split(",")
        rows.append((float(t), float(v), src))
    return rows


def main() -> None:
    omega0 = read_canonical("omega0_T")
    rel = read_canonical("rel_damping_T")
    gabs = read_canonical("gamma_abs_T")

    # interpolable omega_s per source for the rel->absolute conversion
    by_src = {}
    for t, v, s in omega0:
        by_src.setdefault(s, []).append((t, v))
    import numpy as np
    interp = {}
    for s, pts in by_src.items():
        pts.sort()
        interp[s] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    # rel-damping sources map onto the matching omega0 sources
    rel_to_om = {"vsr1982_fig3_thiswork": "vsr1982_fig3_thiswork",
                 "luspin1980_via_vsr_fig3": "luspin1980_via_vsr_fig3",
                 "barker1966_via_vsr_fig3": "barker1966_via_vsr_fig3"}

    lines = [
        "# softmode_inputs_BaTiO3.csv - COMPLETE zone-center soft-mode inputs, cubic BaTiO3",
        "# Rebuilt by scripts/build_bto_softmode_inputs.py from the canonical merged",
        "# digitizations in data/processed/bto_softmode_digitized/ (VSR 1982 Fig. 3+6,",
        "# Presting 1983 Fig. 2; 4/4 consistency checks passed) plus the retained",
        "# Harada 1971 finite-q neutron anchors and Ehsan 2021 SCP anchors from the",
        "# 2026-07-23 import. Conventions applied in code, never in raw files:",
        "# Gamma_HWHM = gamma_full/2 (classical-oscillator FULL damping halved);",
        "# overdamped_flag = 1 where gamma_full/Omega0 > 1. All T in Kelvin (axis",
        "# units verified against the figure images; see references/digitized/*/README.md).",
        "# omega_s(T) is to be interpolated from the MEASURED points — VSR observe",
        "# systematic deviation from the linear Curie-Weiss law, so NO Cochran-linear",
        "# fit is imposed (noted for manuscript section 4.5).",
        "quantity,T_K,value_cm1,overdamped_flag,source",
    ]

    for t, v, s in omega0:
        lines.append(f"omega_s,{t:.2f},{v:.4g},,{s}")
    n_od = 0
    for t, rd, s in rel:
        om = float(np.interp(t, *interp[rel_to_om[s]]))
        gamma_full = rd * om
        hwhm = gamma_full / 2.0
        od = 1 if rd > 1.0 else 0
        n_od += od
        lines.append(f"Gamma_HWHM,{t:.2f},{hwhm:.4g},{od},{s}")
    for t, g, s in gabs:
        om_src = ("vsr1982_via_presting_fig2" if "vsr" in s
                  else "inoue_akimoto1983_via_presting_fig2")
        om = float(np.interp(t, *interp[om_src]))
        od = 1 if g / om > 1.0 else 0
        n_od += od
        lines.append(f"Gamma_HWHM,{t:.2f},{g / 2.0:.4g},{od},{s}")

    # retained context anchors (finite-q neutron; SCP theory)
    lines += [
        "omega_softTO_finite_q,723,120.98,,HAS1971_q0.313invA",
        "omega_softTO_finite_q,523,102.43,,HAS1971_q0.313invA",
        "omega_softTO_finite_q,423,79.04,,HAS1971_q0.313invA",
        "omega_softTO_finite_q,423,112.11,,HAS1971_q0.470invA",
        "Gamma_HWHM_finite_q,723,137.11,1,HAS1971_q0.313invA",
        "Gamma_HWHM_finite_q,523,116.95,1,HAS1971_q0.313invA",
        "Gamma_HWHM_finite_q,423,88.72,1,HAS1971_q0.313invA",
        "Gamma_HWHM_finite_q,423,133.89,1,HAS1971_q0.470invA",
        f"A_soft_100_cm2A2,423,{972.0 * MEV_TO_CM1**2:.5g},,HAS1971_dispersion_pm20pct",
        f"A_stiff_110_111_cm2A2,423,{4750.0 * MEV_TO_CM1**2:.5g},,HAS1971_dispersion",
        "omega_scp_Gamma,500,170.02,,EHSAN2021_SCP",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"-> {OUT.relative_to(REPO)} ({len(lines) - 12} data rows; "
          f"{n_od} Gamma rows overdamped-flagged)")


if __name__ == "__main__":
    main()
