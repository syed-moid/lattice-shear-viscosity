#!/usr/bin/env python3
"""Audit of the production eta assembly (hybrid model tut_z_od1, SELF_OFFDIAG = 1, inner mesh 12^3, construction C) at 300 K:
mode-space accounting (every renormalised mode slot summed once, skips reconciled), the bare-omega_0
sector decomposition, the top-20 contributors, and the committed spectral-density export
(data/processed/eta_spectral_density_SrTiO3.csv) that Fig. 1 regenerates from.

The spectral density is binned by the bare omega_0 of each renormalised mode's max-overlap bare
partner (character label only; every frequency, eigenvector and coupling is the production surface's).
Columns: eta_C (production), eta_C_gamma_sector (the Vogt-anchored Gamma TO1 triplet, listed in its
partner bin), eta_B and eta_A (the two other constructions on the same surface, for the ledger).
Usage: uv run python scripts/audit_eta_assembly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_eta_SrTiO3 import (  # noqa: E402
    SECTOR_KEYS, assemble_construction, construction_modes, load_construction_surface, load_vogt,
)

T_K = 300
N_Q = 11 ** 3


def main() -> None:
    vogt, _ = load_vogt()
    surface = load_construction_surface(T_K)
    modes = construction_modes(T_K, 11, surface)
    eta, sectors, flags, details = assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, return_details=True)
    print(f"eta(300 K) = {eta:.4e} Pa s (must reproduce the production run)")

    claimed = {}
    for d in details:
        key = (d["iq"], d["nu"])
        if key in claimed:
            print(f"DOUBLE COUNT: {key} in both {claimed[key]} and {d['sector']}")
        claimed[key] = d["sector"]
    n_total_space = N_Q * 15
    n_acoustic = sum(1 for m in modes if m["iq"] == 0 and m["omega_r"] < 5.0)
    n_summed = len(details)
    print("\nmode-space accounting:")
    print(f"  total (q, nu) space             : {n_total_space}")
    print(f"  skipped: acoustic Gamma transl. : {n_acoustic}")
    print(f"  skipped: non-positive omega_r   : {flags['n_skipped']}")
    for s in SECTOR_KEYS:
        n = sum(1 for d in details if d["sector"] == s)
        print(f"  {s:>14}: {n:6d} modes, eta {sectors[s]:.3e} Pa s ({100 * sectors[s] / eta:5.1f} %)")
    disjoint = len(claimed) == n_summed
    complete = n_acoustic + flags["n_skipped"] + n_summed == n_total_space
    print(f"  disjoint: {'PASS' if disjoint else 'FAIL'}; complete: {'PASS' if complete else 'FAIL'}")
    print(f"  modes with partner overlap < 0.9: {flags['n_low_overlap']} "
          f"({100 * sum(d['eta_contrib'] for d in details if d['overlap_multiplet'] < 0.9) / eta:.1f} % of eta)")

    top = sorted(details, key=lambda d: -d["eta_contrib"])[:20]
    print("\n  top-20 single-mode contributors:")
    print(f"  {'iq':>5} {'nu':>3} {'sector':>11} {'omega0':>8} {'omega_r':>8} {'ovl':>5} {'Gamma':>7} {'gamma_C':>8} {'tau ps':>8} {'eta_i':>10} {'%':>5}")
    for d in top:
        print(f"  {d['iq']:5d} {d['nu']:3d} {d['sector']:>11} {d['omega0']:8.2f} {d['omega_r']:8.2f} {d['overlap_multiplet']:5.2f} "
              f"{d['gamma_hwhm']:7.2f} {d['gruneisen']:8.2f} {d['tau_s'] * 1e12:8.3f} {d['eta_contrib']:10.3e} {100.0 * d['eta_contrib'] / eta:5.1f}")
    print(f"  top-20 share of eta: {100.0 * sum(d['eta_contrib'] for d in top) / eta:.1f} %")
    export_spectral_density(modes, surface, vogt, eta, details)


def export_spectral_density(modes=None, surface=None, vogt=None, eta=None, details=None) -> None:
    """Committed-CSV export of the mode-resolved eta density at 300 K (Fig. 1's data source)."""
    if surface is None:
        vogt, _ = load_vogt()
        surface = load_construction_surface(T_K)
        modes = construction_modes(T_K, 11, surface)
        eta, _, _, details = assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, return_details=True)
    det = {"C": details}
    for cons in ("B", "A"):
        _, _, _, det[cons] = assemble_construction(T_K, construction=cons, modes=modes, surface=surface, vogt=vogt, return_details=True)
    edges = np.arange(-100.0, 925.0, 25.0)
    out = ["omega0_lo_cm1,omega0_hi_cm1,eta_C_Pas,eta_C_gamma_sector_Pas,eta_B_Pas,eta_A_Pas"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = sum(d["eta_contrib"] for d in det["C"] if lo <= d["omega0"] < hi and d["sector"] != "gamma_sector")
        g = sum(d["eta_contrib"] for d in det["C"] if lo <= d["omega0"] < hi and d["sector"] == "gamma_sector")
        b = sum(d["eta_contrib"] for d in det["B"] if lo <= d["omega0"] < hi)
        a = sum(d["eta_contrib"] for d in det["A"] if lo <= d["omega0"] < hi)
        out.append(f"{lo:.0f},{hi:.0f},{c:.6e},{g:.6e},{b:.6e},{a:.6e}")
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "eta_spectral_density_SrTiO3.csv"
    header = [
        "# eta_spectral_density_SrTiO3.csv - produced by scripts/audit_eta_assembly.py",
        f"# mode-resolved eta_xyxy contribution at T = {T_K} K binned by the bare omega_0 (25 cm-1 bins) of each",
        "# renormalised mode's max-overlap bare partner; production model = hybrid model tut_z_od1 (SELF_OFFDIAG = 1, inner mesh 12^3),",
        f"# construction C (total eta_C = {eta:.4e} Pa s); eta_C_gamma_sector = the Vogt-anchored Gamma TO1 triplet",
        "# (listed in its partner bin, not included in eta_C_Pas); eta_B, eta_A = the other constructions on the",
        "# same surface (ledger rows). Negative omega0 = bare-imaginary partner manifold.",
    ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(out) + "\n")
    tmp.rename(path)
    tot = sum(float(line.split(",")[2]) + float(line.split(",")[3]) for line in out[1:])
    print(f"-> {path.name} (sum check: {tot:.4e} vs eta {eta:.4e})")


if __name__ == "__main__":
    main()
