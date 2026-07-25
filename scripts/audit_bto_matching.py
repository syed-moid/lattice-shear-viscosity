#!/usr/bin/env python3
"""Artifact-vs-physical audit of BaTiO3's zone-wide nonlinearity failure.

BaTiO3's shear +/-0.5% curvature diagnostic FAILED zone-wide (47% flag
rate even at omega0 >= 200 cm-1, no clean cutoff up to 300 cm-1 — see
GATE_1p.md, 2026-07-23 entry), unlike SrTiO3 where flags collapse above
175 cm-1. Before any Route-H-everywhere decision, this audit rules a
matcher/pairing artifact in or out. Four independent probes:

(a) Overlap-score distribution of the ACTUAL matched pairs, stratified
    by reference omega0 (imaginary / near-zero / low / mid / high) —
    ambiguous pairing shows up as low matched-overlap scores. Also counts
    stable-reference modes whose matched strained partner is NEGATIVE
    (imaginary under strain): those enter the omega^2 fits with the wrong
    sign (matched**2 = +|omega|^2, but the physical dynamical-matrix
    eigenvalue is -|omega|^2), a genuine artifact channel near the
    stability boundary. (Note read_modes itself is clean: matdyn's
    negative-frequency convention passes through with no abs()/sqrt.)

(b) Flag rate stratified two ways: by omega0 bins (same as SrTiO3), and
    by distance-to-instability — per-q (min reference frequency at that
    q-point) and per-branch-index (min reference frequency of that branch
    index across the zone; branch index is a proxy since matdyn orders by
    frequency, noted in the output). Physical near-singular-surface
    failure should correlate with instability proximity; a flat/random
    flag rate in these coordinates points at the matcher.

(c) Synthetic sign-flip/degeneracy tests run against ACTUAL BaTiO3
    reference eigenvectors (complex, off-Gamma) at q-points on the
    Gamma-X and Gamma-M lines where the flat imaginary TO branches live:
    build a synthetic linear perturbation on each degenerate subspace
    with eigenvalues spanning both signs (the historical sign-flip
    pathology trigger), generate exact +/-eps synthetic data, and check
    the matcher recovers zero curvature. Matcher failures show up as
    nonzero a2 on the synthetic (exactly-linear) input.

(d) The same curvature metrics on the hydrostatic +/-0.5% pair.
    Hydrostatic strain preserves cubic symmetry (degenerate subspaces do
    not split), so mode-tracking is far easier: if hydro is clean while
    shear fails, suspicion falls on pairing under symmetry-breaking
    strain; if hydro also fails zone-wide, the physical reading (broad
    near-singular anharmonic surface) is supported.

Reads : data/raw/gruneisen_modes/BaTiO3/{reference,shear_xy_*,hydro_*}.modes
        data/raw/gruneisen_modes/SrTiO3/... (control comparisons)
Writes: data/processed/reports/audit_bto_matching.csv

Usage: uv run python scripts/audit_bto_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import H, MASSES, MODES_DIR, FLAG_FRACTION, compute_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import (  # noqa: E402
    _degenerate_groups, _pair_by_mutual_overlap, orthonormal_eigenvectors,
)
from latvisc.qe_modes import read_modes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEGEN_TOL = 0.5
OMEGA_MIN = 5.0

OMEGA_CATS = [("imaginary (<0)", -np.inf, 0.0), ("near-zero [0,5)", 0.0, 5.0),
              ("[5,50)", 5.0, 50.0), ("[50,200)", 50.0, 200.0), ("[200,inf)", 200.0, np.inf)]


def matched_pair_details(directory: Path, masses, plus_name: str, minus_name: str):
    """Run the pairing machinery keeping per-mode matched-overlap scores.

    Returns list of per-(q,branch) dicts: omega_ref, matched_plus/minus
    frequency, matched overlap score vs reference for each sign, and the
    q-point's min reference frequency.
    """
    reference = read_modes(directory / "reference.modes")
    plus = read_modes(directory / f"{plus_name}.modes")
    minus = read_modes(directory / f"{minus_name}.modes")
    out = []
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        _, freq_p, vec_p = plus[iq]
        _, freq_m, vec_m = minus[iq]
        z_ref = orthonormal_eigenvectors(vec_ref, masses)
        z_p = orthonormal_eigenvectors(vec_p, masses)
        z_m = orthonormal_eigenvectors(vec_m, masses)
        freq_ref = np.asarray(freq_ref, dtype=float)
        freq_p = np.asarray(freq_p, dtype=float)
        freq_m = np.asarray(freq_m, dtype=float)
        groups = _degenerate_groups(freq_ref, DEGEN_TOL)
        chosen_p, chosen_m = _pair_by_mutual_overlap(
            freq_ref, z_ref, freq_p, z_p, freq_m, z_m, groups)
        ov_p = np.abs(z_ref.conj() @ z_p.T)
        ov_m = np.abs(z_ref.conj() @ z_m.T)
        qmin = float(freq_ref.min())
        for b in range(len(freq_ref)):
            out.append({
                "iq": iq, "branch": b + 1, "omega_ref": float(freq_ref[b]),
                "matched_p": float(freq_p[chosen_p[b]]),
                "matched_m": float(freq_m[chosen_m[b]]),
                "score_p": float(ov_p[b, chosen_p[b]]),
                "score_m": float(ov_m[b, chosen_m[b]]),
                "q_min_freq": qmin,
            })
    return out


def probe_a(material: str) -> None:
    print(f"\n=== (a) {material}: matched-overlap scores + negative-partner census (shear pair) ===")
    details = matched_pair_details(MODES_DIR / material, MASSES[material],
                                  "shear_xy_p005", "shear_xy_m005")
    scores = np.array([min(d["score_p"], d["score_m"]) for d in details])
    omegas = np.array([d["omega_ref"] for d in details])
    print(f"{'omega0 category':>18} {'n':>6} {'score med':>10} {'score p10':>10} {'score min':>10}")
    for label, lo, hi in OMEGA_CATS:
        mask = (omegas >= lo) & (omegas < hi)
        if mask.sum() == 0:
            continue
        s = scores[mask]
        print(f"{label:>18} {int(mask.sum()):6d} {np.median(s):10.4f} "
              f"{np.percentile(s, 10):10.4f} {s.min():10.4f}")

    stable = [d for d in details if d["omega_ref"] >= OMEGA_MIN]
    neg_partner = [d for d in stable if d["matched_p"] < 0 or d["matched_m"] < 0]
    print(f"stable-reference modes with a NEGATIVE (imaginary) matched strained partner: "
          f"{len(neg_partner)}/{len(stable)}")
    for d in neg_partner[:10]:
        print(f"  iq={d['iq']} br={d['branch']} omega0={d['omega_ref']:.2f} "
              f"p={d['matched_p']:.2f} m={d['matched_m']:.2f} "
              f"scores=({d['score_p']:.3f},{d['score_m']:.3f})")
    if len(neg_partner) > 10:
        print(f"  ... and {len(neg_partner) - 10} more")
    return details


def probe_b(material: str) -> None:
    print(f"\n=== (b) {material}: flag rate vs |omega0| and vs distance-to-instability ===")
    rows = compute_dataset(MODES_DIR / material, MASSES[material], mesh_n=11)
    stable = [r for r in rows if not r["unstable"]]
    gxy2 = np.array([r["gamma_omega2"] for r in stable
                     if not r["acoustic"] and np.isfinite(r["gamma_omega2"])])
    rms2 = float(np.sqrt(np.mean(gxy2 ** 2)))

    def is_flagged(r):
        return (np.isfinite(r["gamma_omega2"]) and rms2 > 0
                and abs(r["b2"] * H / r["omega_ref"] ** 2) / rms2 > FLAG_FRACTION)

    # per-q min reference frequency (instability depth at that q)
    qmin = {}
    for r in rows:
        qmin[r["iq"]] = min(qmin.get(r["iq"], np.inf), r["omega_ref"])
    # per-branch-index min frequency across the zone (proxy: matdyn orders
    # by frequency at each q, so branch index is not exact band identity)
    bmin = {}
    for r in rows:
        bmin[r["branch"]] = min(bmin.get(r["branch"], np.inf), r["omega_ref"])

    print("  by |omega0| (same bins as SrTiO3):")
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, np.inf)]:
        sub = [r for r in stable if lo <= r["omega_ref"] < hi]
        if not sub:
            continue
        fl = sum(1 for r in sub if is_flagged(r))
        print(f"    [{lo:>4},{hi if hi < np.inf else 'inf':>4}): {fl}/{len(sub)} "
              f"({100.0 * fl / len(sub):5.1f}%)")

    print("  by q-point instability depth (min reference freq at that q, cm-1):")
    qbins = [(-np.inf, -100), (-100, -50), (-50, -5), (-5, 5), (5, np.inf)]
    for lo, hi in qbins:
        sub = [r for r in stable if lo <= qmin[r["iq"]] < hi]
        if not sub:
            continue
        fl = sum(1 for r in sub if is_flagged(r))
        label = f"[{lo if lo > -np.inf else '-inf'},{hi if hi < np.inf else 'inf'})"
        print(f"    q_min_freq in {label:>12}: {fl}/{len(sub)} "
              f"({100.0 * fl / len(sub):5.1f}%)")

    print("  same, but only for HIGH-omega0 modes (>=200 cm-1) — does high-omega")
    print("  flagging track the q-point's instability, or is it flat?")
    for lo, hi in qbins:
        sub = [r for r in stable if r["omega_ref"] >= 200 and lo <= qmin[r["iq"]] < hi]
        if not sub:
            continue
        fl = sum(1 for r in sub if is_flagged(r))
        label = f"[{lo if lo > -np.inf else '-inf'},{hi if hi < np.inf else 'inf'})"
        print(f"    q_min_freq in {label:>12}: {fl}/{len(sub)} "
              f"({100.0 * fl / len(sub):5.1f}%)")

    print("  by branch-index zone-minimum frequency (band-identity proxy):")
    for lo, hi in [(-np.inf, -50), (-50, 0), (0, 50), (50, np.inf)]:
        sub = [r for r in stable if lo <= bmin[r["branch"]] < hi]
        if not sub:
            continue
        fl = sum(1 for r in sub if is_flagged(r))
        label = f"[{lo if lo > -np.inf else '-inf'},{hi if hi < np.inf else 'inf'})"
        print(f"    branch_min_freq in {label:>12}: {fl}/{len(sub)} "
              f"({100.0 * fl / len(sub):5.1f}%)")
    return rows, rms2


def probe_c(material: str, rng_seed: int = 7) -> None:
    """Synthetic exactly-linear perturbation on REAL reference eigenvectors
    at Gamma-X / Gamma-M q-points; matcher must recover zero curvature."""
    print(f"\n=== (c) {material}: synthetic sign-flip tests on real eigenvectors "
          f"(Gamma-X / Gamma-M lines) ===")
    reference = read_modes(MODES_DIR / material / "reference.modes")
    masses = MASSES[material]
    rng = np.random.default_rng(rng_seed)
    h = H

    # locate q on the GX (x,0,0) and GM (x,x,0) lines (fractional-derived
    # Cartesian printout; use pattern of components instead of exact values)
    def on_lines(q):
        ax = np.abs(q)
        gx = (ax[1] < 1e-6) and (ax[2] < 1e-6) and (ax[0] > 1e-6)
        gm = (abs(ax[0] - ax[1]) < 1e-6) and (ax[2] < 1e-6) and (ax[0] > 1e-6)
        return gx or gm

    n_q_tested = n_groups = n_fail = 0
    worst = 0.0
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        if not on_lines(np.asarray(q)):
            continue
        n_q_tested += 1
        freq_ref = np.asarray(freq_ref, dtype=float)
        z_ref = orthonormal_eigenvectors(vec_ref, masses)
        groups = _degenerate_groups(freq_ref, DEGEN_TOL)

        # synthetic strained data: per degenerate group of size>=2, rotate
        # the subspace by a random unitary (the perturbation eigenbasis) and
        # split frequencies by symmetric +/- eigenvalues (sign-flip trigger);
        # non-degenerate modes shift linearly with a random slope.
        freq_p = freq_ref.copy()
        freq_m = freq_ref.copy()
        z_p = z_ref.copy()
        z_m = z_ref.copy()
        for g in groups:
            g = np.array(g)
            if len(g) == 1:
                slope = rng.uniform(-2000, 2000)
                freq_p[g] = freq_ref[g] + slope * h
                freq_m[g] = freq_ref[g] - slope * h
                continue
            n_groups += 1
            k = len(g)
            # random unitary in the subspace via QR of a complex Gaussian
            a = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
            u, _ = np.linalg.qr(a)
            z_new = u.T @ z_ref[g]           # perturbation eigenbasis
            lams = np.linspace(-1500, 1500, k)  # eigenvalues spanning both signs
            base = freq_ref[g].mean()
            z_p[g] = z_new
            z_m[g] = z_new                   # sign-independent eigenvectors
            freq_p[g] = base + lams * h
            freq_m[g] = base - lams * h

        # scramble the printed order of the strained modes (matdyn sorts by
        # frequency, so emulate that: sort ascending) — the matcher must undo it
        order_p = np.argsort(freq_p)
        order_m = np.argsort(freq_m)
        chosen_p, chosen_m = _pair_by_mutual_overlap(
            freq_ref, z_ref, freq_p[order_p], z_p[order_p],
            freq_m[order_m], z_m[order_m], groups)
        rec_p = freq_p[order_p][chosen_p]
        rec_m = freq_m[order_m][chosen_m]
        # exactly-linear input => midpoint must equal the group-mean reference
        # (within the degeneracy spread) => curvature a2 must vanish
        a2 = (rec_p + rec_m - 2.0 * freq_ref) / (2.0 * h * h)
        # degenerate groups contribute |freq_ref - group mean|/h^2 spread even
        # on perfect matching; measure against that floor
        floor = np.zeros_like(freq_ref)
        for g in groups:
            g = np.array(g)
            if len(g) > 1:
                floor[g] = np.abs(freq_ref[g] - freq_ref[g].mean()) / (h * h)
        excess = np.abs(a2) - floor - 1e-6 / (h * h)  # printed-precision allowance
        bad = excess > 0
        n_fail += int(bad.sum())
        worst = max(worst, float(np.max(np.abs(a2) - floor)))
    print(f"  q-points tested on GX/GM lines: {n_q_tested}; degenerate groups "
          f"synthesized: {n_groups}")
    print(f"  branch-slots with curvature above the degeneracy-spread floor: "
          f"{n_fail} (worst excess {worst:.3g} cm-1/strain^2)")
    verdict = "PASS — matcher recovers exactly-linear synthetic data on real vectors" \
        if n_fail == 0 else "FAIL — matcher artifacts on synthetic linear input"
    print(f"  {verdict}")


def probe_d(material: str) -> None:
    """Hydro-pair curvature with the same omega^2-basis metrics as shear."""
    print(f"\n=== (d) {material}: hydrostatic +/-0.5% pair curvature (symmetry-preserving control) ===")
    directory = MODES_DIR / material
    masses = MASSES[material]
    reference = read_modes(directory / "reference.modes")
    plus = read_modes(directory / "hydro_p005.modes")
    minus = read_modes(directory / "hydro_m005.modes")

    from latvisc.gruneisen import match_strain_pair_by_overlap
    rows = []
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        _, freq_p, vec_p = plus[iq]
        _, freq_m, vec_m = minus[iq]
        matched_p, matched_m = match_strain_pair_by_overlap(
            freq_ref, vec_ref, freq_p, vec_p, freq_m, vec_m, masses, DEGEN_TOL)
        freq_ref = np.asarray(freq_ref, dtype=float)
        omega_safe = np.where(freq_ref > 0, freq_ref, np.nan)
        D = (matched_p ** 2 - matched_m ** 2) / (2.0 * H)
        b2 = (matched_p ** 2 + matched_m ** 2 - 2.0 * freq_ref ** 2) / (2.0 * H ** 2)
        gamma2 = -D / (2.0 * omega_safe ** 2)
        acoustic_q = np.linalg.norm(np.asarray(q)) < 1e-3
        for b in range(len(freq_ref)):
            rows.append({
                "omega_ref": float(freq_ref[b]),
                "b2": float(b2[b]), "gamma_omega2": float(gamma2[b]),
                "unstable": bool(freq_ref[b] < OMEGA_MIN),
                "acoustic": bool(acoustic_q and abs(freq_ref[b]) < OMEGA_MIN),
            })

    stable = [r for r in rows if not r["unstable"]]
    g2 = np.array([r["gamma_omega2"] for r in stable
                   if not r["acoustic"] and np.isfinite(r["gamma_omega2"])])
    rms2 = float(np.sqrt(np.mean(g2 ** 2)))
    mean2 = float(np.mean(g2))
    print(f"  <gamma_omega2(hydro)> = {mean2:.4f} (volumetric gamma/3-like slope; "
          f"NONZERO by symmetry, unlike shear), rms = {rms2:.4f}")
    print("  flag rate |b2*h/omega0^2| / rms(gamma_omega2_hydro) > 10%, by omega0 bin:")
    total_fl = 0
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, np.inf)]:
        sub = [r for r in stable if lo <= r["omega_ref"] < hi]
        if not sub:
            continue
        fl = sum(1 for r in sub
                 if np.isfinite(r["gamma_omega2"]) and rms2 > 0
                 and abs(r["b2"] * H / r["omega_ref"] ** 2) / rms2 > FLAG_FRACTION)
        total_fl += fl
        print(f"    [{lo:>4},{hi if hi < np.inf else 'inf':>4}): {fl}/{len(sub)} "
              f"({100.0 * fl / len(sub):5.1f}%)")
    print(f"  overall hydro flag rate: {total_fl}/{len(stable)} "
          f"({100.0 * total_fl / len(stable):.1f}%)")


def main() -> None:
    material = "BaTiO3"
    ref = read_modes(MODES_DIR / material / "reference.modes")
    n_neg = sum(int(np.sum(np.asarray(f) < 0)) for _, f, _ in ref)
    n_tot = sum(len(f) for _, f, _ in ref)
    n_q_unstable = sum(1 for _, f, _ in ref if np.asarray(f).min() < 0)
    print(f"{material} reference: {n_neg}/{n_tot} (q,branch) entries imaginary "
          f"({100.0 * n_neg / n_tot:.1f}%), {n_q_unstable}/{len(ref)} q-points "
          f"contain >=1 unstable branch")

    probe_a(material)
    probe_b(material)
    probe_c(material)
    probe_d(material)

    # SrTiO3 shear control for probe (a)'s score distribution
    print("\n=== control: SrTiO3 shear matched-overlap scores (same probe as (a)) ===")
    details_sto = matched_pair_details(MODES_DIR / "SrTiO3", MASSES["SrTiO3"],
                                      "shear_xy_p005", "shear_xy_m005")
    scores = np.array([min(d["score_p"], d["score_m"]) for d in details_sto])
    omegas = np.array([d["omega_ref"] for d in details_sto])
    print(f"{'omega0 category':>18} {'n':>6} {'score med':>10} {'score p10':>10} {'score min':>10}")
    for label, lo, hi in OMEGA_CATS:
        mask = (omegas >= lo) & (omegas < hi)
        if mask.sum() == 0:
            continue
        s = scores[mask]
        print(f"{label:>18} {int(mask.sum()):6d} {np.median(s):10.4f} "
              f"{np.percentile(s, 10):10.4f} {s.min():10.4f}")
    stable = [d for d in details_sto if d["omega_ref"] >= OMEGA_MIN]
    neg = sum(1 for d in stable if d["matched_p"] < 0 or d["matched_m"] < 0)
    print(f"SrTiO3 stable-reference modes with a negative matched partner: {neg}/{len(stable)}")


if __name__ == "__main__":
    main()
