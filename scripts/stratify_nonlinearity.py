#!/usr/bin/env python3
"""Stratify the existing curvature diagnostic by reference frequency omega0.

No new compute — reads the already-downloaded .modes files (same inputs
check_shear_nonlinearity.py uses) and re-derives per-mode quantities
locally. Tests the hypothesis that the residual nonlinearity found in
check_shear_nonlinearity.py (see the G1' incident log,
data/processed/reports/GATE_1p.md) concentrates in the low-omega0 modes
near the lattice instabilities, where the bare cubic-harmonic surface is
already a poor description — rather than being uniform across the zone.

Also confirms the omega^2-native flag metric used in
check_shear_nonlinearity.py is well-posed: |b2*h/omega_ref^2| (the
curvature-induced perturbation to gamma_omega2) compared against
rms(gamma_omega2) over the BZ, never the ratio to D (which — like the
original a1-denominator ratio — blows up wherever gamma passes through
zero, a large fraction of the BZ by symmetry, and those modes contribute
~nothing to eta ~ gamma^2).

If flags and Delta_eta/eta collapse onto the low-omega0 bins, this
proposes a data-driven omega0 cutoff for a Route-S/Route-H partition:
eps05 central differences stand for the high-omega0 manifold (if its bins
are clean), while the near-instability region is excluded from bare-
harmonic gamma and routed through the soft-mode treatment,
gamma = Lambda/(2*omega^2(T)) with renormalized frequencies, Lambda taken
from these same strained cells (Lambda = D = d(omega^2)/d(strain)).

Usage: uv run python scripts/stratify_nonlinearity.py <material>
Writes: data/processed/reports/nonlinearity_stratified_<material>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import (  # noqa: E402
    ABS_FLAG_THRESHOLD_GAMMA_OMEGA2, FLAG_FRACTION, H, MASSES, MODES_DIR,
    TEMPERATURE_K, compute_dataset,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.viscosity import bose_einstein  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CM1_TO_RAD_PER_S = 2.0 * np.pi * speed_of_light * 100.0
OMEGA_BINS = [(0, 50), (50, 100), (100, 200), (200, np.inf)]


def mode_eta_contribution(rows, gamma_key: str, delta_key_fn):
    """Per-mode (weight, gamma, delta_gamma, eta0_term, delta_eta_plus_term,
    delta_eta_minus_term) for stable, non-acoustic modes."""
    usable = [r for r in rows if not r["unstable"] and not r["acoustic"]
              and np.isfinite(r[gamma_key])]
    omega_rad = np.array([r["omega_ref"] for r in usable]) * CM1_TO_RAD_PER_S
    gamma = np.array([r[gamma_key] for r in usable])
    delta_gamma = np.array([delta_key_fn(r) for r in usable])
    n_occ = bose_einstein(omega_rad, TEMPERATURE_K)
    weight = omega_rad ** 2 * n_occ * (n_occ + 1.0)
    eta0_term = weight * gamma ** 2
    delta_plus_term = weight * (gamma + delta_gamma) ** 2 - eta0_term
    delta_minus_term = weight * (gamma - delta_gamma) ** 2 - eta0_term
    return usable, eta0_term, delta_plus_term, delta_minus_term


def stratify(material: str) -> None:
    masses = MASSES[material]
    directory = MODES_DIR / material
    rows = compute_dataset(directory, masses, mesh_n=11)
    stable = [r for r in rows if not r["unstable"]]

    gxy = np.array([r["gamma_xy"] for r in stable if not r["acoustic"]])
    rms_gxy = float(np.sqrt(np.mean(gxy ** 2)))
    gxy2 = np.array([r["gamma_omega2"] for r in stable
                      if not r["acoustic"] and np.isfinite(r["gamma_omega2"])])
    rms_gxy2 = float(np.sqrt(np.mean(gxy2 ** 2)))

    usable_om, eta0_om, dplus_om, dminus_om = mode_eta_contribution(
        rows, "gamma_xy", lambda r: r["a2"] * H / r["omega_ref"]
    )
    usable_om2, eta0_om2, dplus_om2, dminus_om2 = mode_eta_contribution(
        rows, "gamma_omega2", lambda r: r["b2"] * H / r["omega_ref"] ** 2
    )
    eta0_total_om = np.sum(eta0_om)
    eta0_total_om2 = np.sum(eta0_om2)

    print(f"{material}: omega^2 flag metric is |b2*h/omega_ref^2| vs the ABSOLUTE "
          f"threshold {ABS_FLAG_THRESHOLD_GAMMA_OMEGA2:.4f} (SrTiO3-Richardson-calibrated, "
          f"identical for all materials, see check_shear_nonlinearity.py; this material's "
          f"own rms(gamma_omega2)={rms_gxy2:.4f} is printed for context only — a "
          f"per-material rms normalizer misleads across materials, GATE_1p.md 2026-07-23), "
          f"NOT the ratio to D "
          f"(which is ill-posed at gamma zero-crossings, same failure mode as "
          f"the original |a2*h/a1|). Confirmed by code inspection of "
          f"report_flags_omega2 in check_shear_nonlinearity.py.")
    print(f"{material}: eta_44 kernel-weighted propagation confirmed "
          f"((hbar*omega)^2*n(n+1) weight, tau assumed mode-independent) "
          f"— not a per-mode ratio.\n")

    print(f"{material}: stratified by reference frequency omega0 (off-grid 11x11x11 mesh)")
    header = ("omega0_bin_cm1,n_stable,flag_rate_omega_pct,flag_rate_omega2_pct,"
              "eta0_share_pct,delta_eta_over_eta_omega_plus_pct,"
              "delta_eta_over_eta_omega2_plus_pct")
    csv_rows = [header]
    for lo, hi in OMEGA_BINS:
        bin_label = f"[{lo:.0f},{hi if hi < np.inf else 'inf'})"
        bin_stable = [r for r in stable if lo <= r["omega_ref"] < hi]
        n = len(bin_stable) or 1

        flagged_om = sum(1 for r in bin_stable
                          if np.isfinite(r["a2"]) and rms_gxy > 0
                          and abs(r["a2"] * H / r["omega_ref"]) / rms_gxy > FLAG_FRACTION)
        flagged_om2 = sum(1 for r in bin_stable
                           if np.isfinite(r["gamma_omega2"])
                           and abs(r["b2"] * H / r["omega_ref"] ** 2) > ABS_FLAG_THRESHOLD_GAMMA_OMEGA2)

        bin_mask_om = np.array([lo <= r["omega_ref"] < hi for r in usable_om])
        bin_mask_om2 = np.array([lo <= r["omega_ref"] < hi for r in usable_om2])
        eta0_share = 100.0 * np.sum(eta0_om[bin_mask_om]) / eta0_total_om if eta0_total_om > 0 else float("nan")
        delta_om_pct = 100.0 * np.sum(dplus_om[bin_mask_om]) / eta0_total_om if eta0_total_om > 0 else float("nan")
        delta_om2_pct = 100.0 * np.sum(dplus_om2[bin_mask_om2]) / eta0_total_om2 if eta0_total_om2 > 0 else float("nan")

        print(f"  {bin_label:>12} cm-1: n={len(bin_stable):6d}  "
              f"flag(omega)={100.0 * flagged_om / n:5.1f}%  "
              f"flag(omega^2)={100.0 * flagged_om2 / n:5.1f}%  "
              f"eta0 share={eta0_share:6.2f}%  "
              f"Delta_eta/eta contrib(omega)={delta_om_pct:+7.2f}%  "
              f"Delta_eta/eta contrib(omega^2)={delta_om2_pct:+7.2f}%")
        csv_rows.append(f"{bin_label},{len(bin_stable)},{100.0 * flagged_om / n:.2f},"
                         f"{100.0 * flagged_om2 / n:.2f},{eta0_share:.4f},"
                         f"{delta_om_pct:.4f},{delta_om2_pct:.4f}")

    # worst 1% of modes by |contribution to Delta eta| in omega^2 space
    contrib = np.abs(dplus_om2)
    order = np.argsort(contrib)[::-1]
    n_worst = max(1, int(0.01 * len(contrib)))
    worst_share = 100.0 * np.sum(contrib[order[:n_worst]]) / np.sum(contrib) if np.sum(contrib) > 0 else float("nan")
    print(f"\n{material}: worst 1% of modes ({n_worst}/{len(contrib)}) by |Delta_eta contribution| "
          f"(omega^2-space) carry {worst_share:.1f}% of the total |Delta_eta| signed sum")

    # --- task 2: data-driven omega0 cutoff for a Route-S/Route-H partition ---
    print(f"\n{material}: scanning omega0 cutoffs for a Route-S (bare eps05, high-omega0) / "
          f"Route-H (soft-mode Lambda/2*omega^2(T), low-omega0) partition")
    print(f"  {'cutoff':>8}  {'n_above':>8}  {'flag(omega2)_above_pct':>22}  "
          f"{'delta_eta_above_pct':>20}  {'n_below':>8}  {'eta0_below_pct':>15}")
    candidates = list(range(25, 325, 25))
    proposed_cutoff = None
    cutoff_rows = ["cutoff_cm1,n_above,flag_omega2_above_pct,delta_eta_above_pct,n_below,eta0_below_pct"]
    for cutoff in candidates:
        above = [r for r in stable if r["omega_ref"] >= cutoff]
        below = [r for r in stable if r["omega_ref"] < cutoff]
        n_above = len(above) or 1
        flagged_above = sum(1 for r in above
                             if np.isfinite(r["gamma_omega2"])
                             and abs(r["b2"] * H / r["omega_ref"] ** 2) > ABS_FLAG_THRESHOLD_GAMMA_OMEGA2)
        flag_pct_above = 100.0 * flagged_above / n_above

        mask_above = np.array([r["omega_ref"] >= cutoff for r in usable_om2])
        mask_below = np.array([r["omega_ref"] < cutoff for r in usable_om2])
        delta_above_pct = (100.0 * np.sum(dplus_om2[mask_above]) / eta0_total_om2
                            if eta0_total_om2 > 0 else float("nan"))
        eta0_below_pct = (100.0 * np.sum(eta0_om2[mask_below]) / eta0_total_om2
                           if eta0_total_om2 > 0 else float("nan"))

        print(f"  {cutoff:6d}  {len(above):8d}  {flag_pct_above:20.1f}%  "
              f"{delta_above_pct:18.2f}%  {len(below):8d}  {eta0_below_pct:13.2f}%")
        cutoff_rows.append(f"{cutoff},{len(above)},{flag_pct_above:.2f},"
                            f"{delta_above_pct:.2f},{len(below)},{eta0_below_pct:.2f}")
        if proposed_cutoff is None and flag_pct_above < 100 * FLAG_FRACTION:
            proposed_cutoff = cutoff

    if proposed_cutoff is not None:
        below_share = None
        for row in cutoff_rows[1:]:
            parts = row.split(",")
            if int(parts[0]) == proposed_cutoff:
                below_share = parts[4]
        print(f"\n{material}: PROPOSED cutoff omega0 >= {proposed_cutoff} cm-1 "
              f"(smallest scanned cutoff where the above-cutoff omega^2-flag rate "
              f"first drops below {100 * FLAG_FRACTION:.0f}%). Below this cutoff, "
              f"{below_share} q,branch entries fall into the candidate Route-H "
              f"(soft/near-instability) manifold.")
    else:
        print(f"\n{material}: NO scanned cutoff (up to {candidates[-1]} cm-1) brought the "
              f"above-cutoff flag rate below {100 * FLAG_FRACTION:.0f}% — the high-omega0 "
              f"manifold does NOT look clean; the partition hypothesis is not supported "
              f"by this data alone, or the true cutoff lies above the scanned range.")

    out = REPO / "data" / "processed" / "reports" / f"nonlinearity_stratified_{material}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = ("# omega0-stratified flag rates and Delta_eta/eta contributions\n"
            "# " + header + "\n" + "\n".join(csv_rows[1:]) + "\n\n"
            "# Route-S/Route-H cutoff scan\n"
            "# " + cutoff_rows[0] + "\n" + "\n".join(cutoff_rows[1:]) + "\n")
    out.write_text(text)
    print(f"\n-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    for material in sys.argv[1:] or ["SrTiO3"]:
        stratify(material)
