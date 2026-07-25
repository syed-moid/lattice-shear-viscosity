#!/usr/bin/env python3
"""Stage-C audit (A2/A3): sector disjointness + low-omega sensitivity table.

A2 — double-count check: enumerate exactly which (q,branch) mode-slots are
claimed by (i) Route S, (ii) Route H stable, (iii) Route H unstable, and
(iv) the Gamma/Vogt soft sector; assert pairwise disjointness and that the
union plus the two documented skip categories (acoustic Gamma translations;
map-negative lam_r) equals the full 11^3 x 15 mode space.

A3 — sensitivity: eta contribution binned by omega0 within the low-omega
sector, and the top-20 single-mode contributors with gamma, tau, omega.

Usage: uv run python scripts/audit_eta_assembly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
from compute_eta_SrTiO3 import N_Q, assemble, load_vogt  # noqa: E402

T_K = 300


def main() -> None:
    rows = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    vogt, _ = load_vogt()
    eta, sectors, flags, details = assemble(T_K, rows, vogt, return_details=True)

    print(f"eta(300 K) = {eta:.4e} Pa s (must reproduce the production run)")

    # ---- A2: disjointness and completeness ----
    claimed = {}
    for d in details:
        key = (d["iq"], d["branch"])
        if key in claimed:
            print(f"DOUBLE COUNT: {key} in both {claimed[key]} and {d['sector']}")
        claimed[key] = d["sector"]
    n_total_space = N_Q * 15
    n_acoustic = sum(1 for r in rows if r["acoustic"])
    n_summed = len(details)
    n_skipped_map = n_total_space - n_acoustic - n_summed
    by_sector = {}
    for d in details:
        by_sector[d["sector"]] = by_sector.get(d["sector"], 0) + 1
    print("\nA2 — mode-space accounting:")
    print(f"  total (q,branch) space          : {n_total_space}")
    print(f"  skipped: acoustic Gamma transl. : {n_acoustic}")
    print(f"  skipped: non-positive mapped w^2: {flags['n_skipped']} "
          f"(reconciled residual {n_skipped_map})")
    for s in ["routeS", "routeH_stable", "routeH_unstable", "gamma_sector"]:
        print(f"  {s:>16}: {by_sector.get(s, 0)}")
    n_claims = len(claimed)
    disjoint = n_claims == n_summed
    complete = n_acoustic + flags["n_skipped"] + n_summed == n_total_space
    print(f"  disjoint (each summed slot claimed once): "
          f"{'PASS' if disjoint else 'FAIL'} ({n_claims} unique / {n_summed} summed)")
    print(f"  complete (skips + summed = space): {'PASS' if complete else 'FAIL'}")

    # ---- A3: low-omega sensitivity ----
    low = [d for d in details if d["sector"] in
           ("routeH_stable", "routeH_unstable", "gamma_sector")]
    print(f"\nA3 — low-omega sector: {len(low)} modes, "
          f"eta contribution {sum(d['eta_contrib'] for d in low):.4e} Pa s")
    print(f"  {'omega0 bin':>14} {'n':>6} {'eta contrib':>12} {'share of total':>14}")
    bins = [(-np.inf, 0), (0, 5), (5, 25), (25, 50), (50, 75), (75, 100),
            (100, 125), (125, 150), (150, 175)]
    for lo, hi in bins:
        sub = [d for d in low if lo <= d["omega0"] < hi]
        if not sub:
            continue
        c = sum(d["eta_contrib"] for d in sub)
        label = f"[{lo if lo > -np.inf else '-inf'},{hi})"
        print(f"  {label:>14} {len(sub):6d} {c:12.3e} {100.0 * c / eta:13.1f}%")

    top = sorted(details, key=lambda d: -d["eta_contrib"])[:20]
    print("\n  top-20 single-mode contributors (whole sum):")
    print(f"  {'iq':>5} {'br':>3} {'sector':>16} {'omega0':>8} {'omega_r':>8} "
          f"{'Gamma':>7} {'gamma':>8} {'tau ps':>8} {'eta_i':>10} {'%':>5}")
    for d in top:
        print(f"  {d['iq']:5d} {d['branch']:3d} {d['sector']:>16} "
              f"{d['omega0']:8.2f} {d['omega_r']:8.2f} {d['gamma_hwhm']:7.2f} "
              f"{d['gruneisen']:8.2f} {d['tau_s'] * 1e12:8.3f} "
              f"{d['eta_contrib']:10.3e} {100.0 * d['eta_contrib'] / eta:5.1f}")
    top20_share = sum(d["eta_contrib"] for d in top) / eta
    print(f"  top-20 share of eta: {100.0 * top20_share:.1f}% "
          f"({'broad-based' if top20_share < 0.5 else 'concentrated'})")


if __name__ == "__main__":
    main()


def export_spectral_density() -> None:
    """Committed-CSV export of the mode-resolved eta density at 300 K
    (fig1's data source — figures must regenerate from committed CSVs)."""
    rows = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    vogt, _ = load_vogt()
    eta, _, _, details = assemble(T_K, rows, vogt, return_details=True)
    edges = np.arange(-100.0, 925.0, 25.0)
    out = ["omega0_lo_cm1,omega0_hi_cm1,eta_routeS_Pas,eta_routeH_stable_Pas,"
           "eta_routeH_unstable_Pas,eta_gamma_sector_Pas"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        by = {}
        for d in details:
            if lo <= d["omega0"] < hi:
                by[d["sector"]] = by.get(d["sector"], 0.0) + d["eta_contrib"]
        out.append(f"{lo:.0f},{hi:.0f},{by.get('routeS', 0):.6e},"
                   f"{by.get('routeH_stable', 0):.6e},"
                   f"{by.get('routeH_unstable', 0):.6e},"
                   f"{by.get('gamma_sector', 0):.6e}")
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / \
        "eta_spectral_density_SrTiO3.csv"
    header = [
        "# eta_spectral_density_SrTiO3.csv - produced by scripts/audit_eta_assembly.py",
        f"# mode-resolved eta_xyxy contribution binned by BARE omega0 (25 cm-1 bins), T = {T_K} K,",
        "# generation-3 assembly (see data/processed/reports/eta_SrTiO3_stageC.md);",
        f"# total eta = {eta:.4e} Pa s; negative omega0 = bare-imaginary (formerly unstable) manifold.",
    ]
    path.write_text("\n".join(header) + "\n" + "\n".join(out) + "\n")
    print(f"-> {path.name} (sum check: "
          f"{sum(float(x) for line in out[1:] for x in line.split(',')[2:]):.4e} vs eta {eta:.4e})")
