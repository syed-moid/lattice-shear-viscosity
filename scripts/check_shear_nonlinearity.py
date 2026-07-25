#!/usr/bin/env python3
"""Free 3-point shear-strain curvature diagnostic (A5' scope-cut substitute).

Given the unstrained reference and the +/-0.5% shear pair, this computes
for every mode:

  * gamma_xy, via the same central-difference formula used everywhere else
    in the pipeline (latvisc.gruneisen.mode_gruneisen_finite_strain), so
    the diagnostic and scripts/compute_gruneisen.py can never disagree on
    the definition of gamma.
  * the standard central-difference linear and curvature coefficients

        a1 = (omega(+h) - omega(-h)) / (2h)                    [linear slope]
        a2 = (omega(+h) + omega(-h) - 2*omega(0)) / (2*h^2)     [curvature]

+/-h branches are paired by latvisc.gruneisen.match_strain_pair_by_overlap,
which handles reference-degenerate subspaces by DIRECT +eps<->-eps mutual
overlap (a bijective max-overlap assignment) rather than independently
sorting each strain sign against the reference — the latter silently swaps
which physical branch sits at a given output position whenever the
degenerate-subspace perturbation eigenvalues span both signs, fabricating
a spurious |eps| kink at eps=0 that a central second difference reads as
large curvature. See the G1' incident log,
data/processed/reports/GATE_1p.md, for the full history of this and the
earlier (separately fixed) Cartesian-vs-fractional q-coordinate bug.

Flag metric: |a2*h/omega_ref| (the curvature-induced perturbation to gamma
at the sampled strain, dimensionless) compared to rms(gamma_xy) over the
full Brillouin zone — NOT the ratio to a1, which is guaranteed to blow up
wherever gamma_xy passes through zero (a large fraction of the BZ by
symmetry); those modes contribute ~nothing to eta (which enters as
gamma^2) and must not dominate the flag count.

Datasets: this script can be pointed at two independent samplings of the
same strain triple —
  * the production 11x11x11 fractional mesh (data/raw/gruneisen_modes/<material>)
  * the exactly 4x4x4-commensurate subset (data/raw/gruneisen_modes/<material>/ongrid4),
    where matdyn returns the DFPT dynamical matrix exactly, with zero
    Fourier-interpolation error from the coarse-to-fine force-constant
    transform
When both are present, flag rates are reported for each separately: if
flags collapse on the exact grid but persist off it, the off-grid
"curvature" is force-constant interpolation noise amplified by 1/h^2, not
a property of the underlying DFPT data.

Also reports, using the full-BZ (11x11x11) dataset:
  * a neighbor-q correlation of a2 along the mesh's natural qz stepping
    (excluding wrap boundaries) — real anharmonic curvature varies
    smoothly along a branch; noise from finite precision / fc truncation
    does not, so a low/negative correlation is evidence for noise.
  * a propagated sensitivity of the physical observable: eta_44 at 300 K
    recomputed with gamma -> gamma +/- (a2*h/omega_ref), using a
    tau-independent kernel weight (hbar*omega)^2 * n(n+1) (single-mode
    lifetimes are not yet available at this stage of the pipeline; this
    assumes tau is uncorrelated with the curvature artifact, which cancels
    it out of the RATIO Delta-eta/eta — an approximation, documented here
    and in the G1' delta report).

Reads : data/raw/gruneisen_modes/<material>/{reference,shear_xy_p005,shear_xy_m005}.modes
        data/raw/gruneisen_modes/<material>/ongrid4/{same}.modes (optional)
Writes: data/processed/reports/nonlinearity_<material>.csv

Usage: uv run python scripts/check_shear_nonlinearity.py <material>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import match_strain_pair_by_overlap, mode_gruneisen_finite_strain  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402
from latvisc.viscosity import bose_einstein  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODES_DIR = REPO / "data" / "raw" / "gruneisen_modes"
H = 0.005
OMEGA_MIN = 5.0  # cm^-1; below this the mode belongs to the unstable manifold
DEGEN_TOL = 0.5  # cm^-1; same convention as scripts/compute_gruneisen.py
ACOUSTIC_SKIP = 1e-3  # |q| below this: acoustic modes are translations
FLAG_FRACTION = 0.10  # |a2*h/omega_ref| / rms(gamma_xy) above this is flagged
# ABSOLUTE omega^2-basis flag threshold (project convention since 2026-07-23,
# user decision recorded in data/processed/reports/GATE_1p.md): a mode is
# flagged when its curvature-induced gamma perturbation |b2*h/omega_ref^2|
# exceeds this value. Calibrated on SrTiO3 — the material where the 5-point
# Richardson analysis validated the eps05 estimate — as FLAG_FRACTION *
# rms(gamma_xy, eps05) over that analysis's 19812 usable mode-slots
# (richardson_5pt_SrTiO3.csv, rms = 4.182536), and applied IDENTICALLY to
# every material. A per-material relative (BZ-rms-normalized) criterion
# misleads across materials: the normalizer depends on how much of the
# large-gamma soft manifold happens to be excluded as unstable (BaTiO3's rms
# came out 3.8x smaller than SrTiO3's for exactly that reason, inflating its
# apparent flag rate ~4x at identical absolute curvature — see the
# 2026-07-23 audit in GATE_1p.md).
ABS_FLAG_THRESHOLD_GAMMA_OMEGA2 = 0.10 * 4.182536
ETA_SENSITIVITY_FAIL = 0.05  # |Delta eta / eta| at or above this fails the pin test
TEMPERATURE_K = 300.0
CM1_TO_RAD_PER_S = 2.0 * np.pi * speed_of_light * 100.0  # omega[rad/s] = this * freq[cm-1]

MASSES = {
    "SrTiO3": np.array([87.62, 47.867, 15.999, 15.999, 15.999]),
    "BaTiO3": np.array([137.327, 47.867, 15.999, 15.999, 15.999]),
}


def degenerate_mask(freq_ref: np.ndarray) -> np.ndarray:
    n = len(freq_ref)
    groups, current = [], [0]
    for i in range(1, n):
        if abs(freq_ref[i] - freq_ref[current[-1]]) < DEGEN_TOL:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    mask = np.zeros(n, dtype=bool)
    for group in groups:
        if len(group) > 1:
            mask[group] = True
    return mask


def compute_dataset(directory: Path, masses: np.ndarray, mesh_n: int):
    """Returns a list of per-(q,branch) dicts for one modes-file triple."""
    reference = read_modes(directory / "reference.modes")
    plus = read_modes(directory / "shear_xy_p005.modes")
    minus = read_modes(directory / "shear_xy_m005.modes")

    rows = []
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        q_p, freq_p, vec_p = plus[iq]
        q_m, freq_m, vec_m = minus[iq]
        # Printed q is matdyn's Cartesian conversion of the shared
        # fractional input using each cell's own strain-distorted
        # reciprocal lattice — expected to differ slightly by strain even
        # for identical fractional labels (see gen_matdyn_mesh_inputs.py).
        assert np.allclose(q, q_p, atol=0.01) and np.allclose(q, q_m, atol=0.01)

        matched_p, matched_m = match_strain_pair_by_overlap(
            freq_ref, vec_ref, freq_p, vec_p, freq_m, vec_m, masses, DEGEN_TOL
        )
        gamma_xy = mode_gruneisen_finite_strain(freq_ref, matched_p, matched_m, H)
        a1 = (matched_p - matched_m) / (2.0 * H)
        a2 = (matched_p + matched_m - 2.0 * freq_ref) / (2.0 * H**2)

        # omega^2 reparameterization: shear couples ~linearly to omega^2
        # (the dynamical-matrix perturbation is linear in strain, and
        # omega^2 IS the eigenvalue), so D = d(omega^2)/d(strain) is the
        # natural linear-response quantity. Even when omega^2(strain) is
        # EXACTLY linear, omega(strain) = sqrt(omega0^2 + D*strain) still
        # has intrinsic sqrt curvature in omega-space,
        # a2_pred = -D^2 / (8*omega0^3), that grows for small omega0 and
        # is NOT anharmonicity. See the G1' incident log.
        omega_ref_safe = np.where(freq_ref > 0, freq_ref, np.nan)
        # signed dynamical-matrix eigenvalue: matdyn prints imaginary
        # frequencies as NEGATIVE reals, so the eigenvalue is
        # sign(omega)*omega^2 — a bare square silently flips the sign for
        # any strained partner that has gone imaginary under strain
        ev_p = np.sign(matched_p) * matched_p**2
        ev_m = np.sign(matched_m) * matched_m**2
        ev_ref = np.sign(freq_ref) * freq_ref**2
        D = (ev_p - ev_m) / (2.0 * H)
        b2 = (ev_p + ev_m - 2.0 * ev_ref) / (2.0 * H**2)
        gamma_omega2 = -D / (2.0 * omega_ref_safe**2)
        a2_pred_sqrt = -D**2 / (8.0 * omega_ref_safe**3)

        degen = degenerate_mask(freq_ref)
        acoustic_q = np.linalg.norm(q) < ACOUSTIC_SKIP

        for branch in range(len(freq_ref)):
            unstable = bool(freq_ref[branch] < OMEGA_MIN)
            is_acoustic = bool(acoustic_q and abs(freq_ref[branch]) < OMEGA_MIN)
            rows.append({
                "iq": iq, "q": q, "branch": branch + 1,
                "omega_ref": float(freq_ref[branch]),
                "gamma_xy": float(gamma_xy[branch]),
                "a1": float(a1[branch]), "a2": float(a2[branch]),
                "D": float(D[branch]), "b2": float(b2[branch]),
                "gamma_omega2": float(gamma_omega2[branch]),
                "a2_pred_sqrt": float(a2_pred_sqrt[branch]),
                "unstable": unstable, "degenerate": bool(degen[branch]),
                "acoustic": is_acoustic,
            })
    return rows


def neighbor_q_correlation(rows, mesh_n: int) -> float:
    """Pearson correlation of a2 between (iq, iq+1) pairs along qz, excluding wraps."""
    by_iq_branch = {(r["iq"], r["branch"]): r["a2"] for r in rows if not r["unstable"]}
    xs, ys = [], []
    for (iq, branch), a2 in by_iq_branch.items():
        if iq % mesh_n == mesh_n - 1:
            continue  # wraps to a new (i, j) line
        neighbor = by_iq_branch.get((iq + 1, branch))
        if neighbor is not None:
            xs.append(a2)
            ys.append(neighbor)
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def eta_sensitivity(rows) -> tuple[float, float]:
    """Delta-eta/eta at 300 K from gamma -> gamma +/- (a2*h/omega_ref).

    tau is assumed mode-independent (real linewidths are not yet available
    at this stage) and cancels out of the ratio; weight is
    (hbar*omega)^2 * n(n+1) only. Returns (delta_plus, delta_minus).
    """
    usable = [r for r in rows if not r["unstable"] and not r["acoustic"]]
    omega_rad = np.array([r["omega_ref"] for r in usable]) * CM1_TO_RAD_PER_S
    gamma = np.array([r["gamma_xy"] for r in usable])
    delta_gamma = np.array([r["a2"] * H / r["omega_ref"] for r in usable])

    n_occ = bose_einstein(omega_rad, TEMPERATURE_K)
    weight = (omega_rad) ** 2 * n_occ * (n_occ + 1.0)  # hbar cancels in the ratio

    eta0 = np.sum(weight * gamma**2)
    eta_plus = np.sum(weight * (gamma + delta_gamma) ** 2)
    eta_minus = np.sum(weight * (gamma - delta_gamma) ** 2)
    return float(eta_plus / eta0 - 1.0), float(eta_minus / eta0 - 1.0)


def report_flags_omega2(rows, rms_gamma_omega2_bz: float, label: str):
    """Omega^2-basis flags against the ABSOLUTE threshold
    ABS_FLAG_THRESHOLD_GAMMA_OMEGA2 (SrTiO3-Richardson-calibrated, identical
    for all materials — see the constant's comment; the earlier per-material
    BZ-rms normalization is retained only as the rms_gamma_omega2_bz argument
    for context printing, no longer used for flagging). Still NOT the ratio
    to D (the omega^2-space analog of |a2*h/a1|), which is exactly as
    ill-posed at gamma zero-crossings as the original a1-denominator ratio
    was, since D and gamma_omega2 vanish together."""
    stable = [r for r in rows if not r["unstable"]]
    degen = [r for r in stable if r["degenerate"]]
    nondegen = [r for r in stable if not r["degenerate"]]

    def flagged(subset):
        out = []
        for r in subset:
            curvature_gamma2 = abs(r["b2"] * H / r["omega_ref"] ** 2)
            if np.isfinite(curvature_gamma2) and curvature_gamma2 > ABS_FLAG_THRESHOLD_GAMMA_OMEGA2:
                out.append((r["iq"], r["branch"], r["omega_ref"],
                            curvature_gamma2 / ABS_FLAG_THRESHOLD_GAMMA_OMEGA2))
        return out

    flag_nondegen = flagged(nondegen)
    flag_degen = flagged(degen)
    n_nd = len(nondegen) or 1
    n_d = len(degen) or 1
    print(f"[{label}] non-degenerate: {len(nondegen)} stable, "
          f"{len(flag_nondegen)} flagged ({100.0 * len(flag_nondegen) / n_nd:.1f}%)")
    print(f"[{label}] degenerate-subspace: {len(degen)} stable, "
          f"{len(flag_degen)} flagged ({100.0 * len(flag_degen) / n_d:.1f}%)")
    return flag_nondegen, flag_degen, len(nondegen), len(degen)


def eta_sensitivity_omega2(rows) -> tuple[float, float]:
    """Delta-eta/eta at 300 K using gamma_omega2 -> gamma_omega2 +/- (b2*h/omega_ref^2)."""
    usable = [r for r in rows if not r["unstable"] and not r["acoustic"] and np.isfinite(r["gamma_omega2"])]
    omega_rad = np.array([r["omega_ref"] for r in usable]) * CM1_TO_RAD_PER_S
    gamma = np.array([r["gamma_omega2"] for r in usable])
    delta_gamma = np.array([r["b2"] * H / r["omega_ref"] ** 2 for r in usable])

    n_occ = bose_einstein(omega_rad, TEMPERATURE_K)
    weight = omega_rad ** 2 * n_occ * (n_occ + 1.0)

    eta0 = np.sum(weight * gamma**2)
    eta_plus = np.sum(weight * (gamma + delta_gamma) ** 2)
    eta_minus = np.sum(weight * (gamma - delta_gamma) ** 2)
    return float(eta_plus / eta0 - 1.0), float(eta_minus / eta0 - 1.0)


def sqrt_model_check(rows, flagged_keys: set) -> None:
    """Task 1b: compare measured a2 (raw-omega curvature) against the
    sqrt-model prediction a2_pred = -D^2/(8*omega0^3) built from each
    mode's own measured D. Reports correlation and the fraction of
    currently-flagged modes explained (residual << the measured signal)."""
    stable = [r for r in rows if not r["unstable"] and np.isfinite(r["a2_pred_sqrt"])]
    measured = np.array([r["a2"] for r in stable])
    predicted = np.array([r["a2_pred_sqrt"] for r in stable])
    finite = np.isfinite(measured) & np.isfinite(predicted)
    corr = float(np.corrcoef(measured[finite], predicted[finite])[0, 1]) if finite.sum() > 1 else float("nan")

    explained = 0
    for r in stable:
        key = (r["iq"], r["branch"])
        if key not in flagged_keys:
            continue
        residual = abs(r["a2"] - r["a2_pred_sqrt"])
        if residual < 0.5 * abs(r["a2"]):
            explained += 1
    frac_explained = explained / len(flagged_keys) if flagged_keys else float("nan")
    print(f"  sqrt-model check: corr(a2_measured, a2_pred_sqrt) = {corr:.4f}; "
          f"of {len(flagged_keys)} currently-flagged modes, {explained} "
          f"({100.0 * frac_explained:.1f}%) have |residual| < 50% of |a2_measured| "
          f"(explained by the sqrt term alone)")


def binned_flag_rate(rows, flagged_keys: set) -> None:
    """Task 1c: flag rate vs omega0 and vs |gamma_xy| — the sqrt hypothesis
    predicts flags concentrated at low omega0 and at |gamma_xy| >~ 40."""
    stable = [r for r in rows if not r["unstable"]]
    omega_bins = [(0, 100), (100, 300), (300, 600), (600, np.inf)]
    print("  flag rate vs omega_ref (cm-1):")
    for lo, hi in omega_bins:
        subset = [r for r in stable if lo <= r["omega_ref"] < hi]
        if not subset:
            continue
        flagged = sum(1 for r in subset if (r["iq"], r["branch"]) in flagged_keys)
        print(f"    [{lo:4.0f},{hi if hi < np.inf else 'inf':>4}) cm-1: "
              f"{flagged}/{len(subset)} flagged ({100.0 * flagged / len(subset):.1f}%)")

    gamma_bins = [(0, 10), (10, 40), (40, 100), (100, np.inf)]
    print("  flag rate vs |gamma_xy|:")
    for lo, hi in gamma_bins:
        subset = [r for r in stable if lo <= abs(r["gamma_xy"]) < hi]
        if not subset:
            continue
        flagged = sum(1 for r in subset if (r["iq"], r["branch"]) in flagged_keys)
        print(f"    [{lo:4.0f},{hi if hi < np.inf else 'inf':>4}): "
              f"{flagged}/{len(subset)} flagged ({100.0 * flagged / len(subset):.1f}%)")


def report_flags(rows, rms_gamma_bz: float, label: str):
    stable = [r for r in rows if not r["unstable"]]
    degen = [r for r in stable if r["degenerate"]]
    nondegen = [r for r in stable if not r["degenerate"]]

    def flagged(subset):
        out = []
        for r in subset:
            curvature_gamma = abs(r["a2"] * H / r["omega_ref"])
            ratio = curvature_gamma / rms_gamma_bz if rms_gamma_bz > 0 else float("nan")
            if np.isfinite(ratio) and ratio > FLAG_FRACTION:
                out.append((r["iq"], r["branch"], r["omega_ref"], ratio))
        return out

    flag_nondegen = flagged(nondegen)
    flag_degen = flagged(degen)
    n_nd = len(nondegen) or 1
    n_d = len(degen) or 1
    print(f"[{label}] non-degenerate: {len(nondegen)} stable, "
          f"{len(flag_nondegen)} flagged ({100.0 * len(flag_nondegen) / n_nd:.1f}%)")
    print(f"[{label}] degenerate-subspace: {len(degen)} stable, "
          f"{len(flag_degen)} flagged ({100.0 * len(flag_degen) / n_d:.1f}%)")
    return flag_nondegen, flag_degen, len(nondegen), len(degen)


def process(material: str) -> None:
    masses = MASSES[material]
    directory = MODES_DIR / material
    offgrid_rows = compute_dataset(directory, masses, mesh_n=11)

    gxy_bz = np.array([r["gamma_xy"] for r in offgrid_rows
                        if not r["unstable"] and not r["acoustic"]])
    mean_gxy = float(np.mean(gxy_bz))
    rms_gxy = float(np.sqrt(np.mean(gxy_bz ** 2)))
    pin_ratio = abs(mean_gxy) / rms_gxy if rms_gxy > 0 else float("nan")

    print(f"{material}: BZ-weighted <gamma_xy>={mean_gxy:.6f}  rms(gamma_xy)={rms_gxy:.6f}  "
          f"|<gamma_xy>|/rms={pin_ratio:.6f}  [{'FAIL' if pin_ratio > 0.01 else 'PASS'} vs 0.01 threshold]")

    fnd_off, fd_off, n_nd_off, n_d_off = report_flags(offgrid_rows, rms_gxy, "off-grid 11x11x11")

    ongrid_dir = directory / "ongrid4"
    ongrid_rows = None
    fnd_on = fd_on = n_nd_on = n_d_on = None
    if (ongrid_dir / "reference.modes").exists():
        ongrid_rows = compute_dataset(ongrid_dir, masses, mesh_n=4)
        fnd_on, fd_on, n_nd_on, n_d_on = report_flags(ongrid_rows, rms_gxy, "on-grid 4x4x4 (exact, no fc interpolation)")

    corr_off = neighbor_q_correlation(offgrid_rows, 11)
    print(f"{material}: neighbor-q correlation of a2 (off-grid 11x11x11): {corr_off:.4f} "
          f"(near 1 = smooth/real, near 0 or negative = ragged/noise)")
    if ongrid_rows is not None:
        corr_on = neighbor_q_correlation(ongrid_rows, 4)
        print(f"{material}: neighbor-q correlation of a2 (on-grid 4x4x4, few points, indicative only): {corr_on:.4f}")

    delta_plus, delta_minus = eta_sensitivity(offgrid_rows)
    worst_eta_sensitivity = max(abs(delta_plus), abs(delta_minus))
    print(f"{material}: eta_44 sensitivity at {TEMPERATURE_K:.0f} K to gamma -> gamma +/- a2*h/omega_ref "
          f"(tau assumed mode-independent): Delta_eta/eta = {delta_plus:+.4f} (+) / {delta_minus:+.4f} (-)  "
          f"worst |Delta_eta/eta| = {worst_eta_sensitivity:.4f}  "
          f"[{'FAIL' if worst_eta_sensitivity >= ETA_SENSITIVITY_FAIL else 'PASS'} vs {ETA_SENSITIVITY_FAIL} threshold]")

    on_grid_flag_rate = (len(fnd_on) / n_nd_on) if ongrid_rows is not None and n_nd_on else float("nan")
    decision_pass_omega = (
        ongrid_rows is not None
        and on_grid_flag_rate < FLAG_FRACTION
        and worst_eta_sensitivity < ETA_SENSITIVITY_FAIL
    )
    print(f"{material}: [omega-space] on-grid non-degenerate flag rate = "
          f"{100.0 * on_grid_flag_rate:.1f}% (threshold {100 * FLAG_FRACTION:.0f}%), "
          f"worst |Delta_eta/eta| = {worst_eta_sensitivity:.4f} (threshold {ETA_SENSITIVITY_FAIL}) "
          f"-> {'PASS' if decision_pass_omega else 'FAIL'}")

    # --- omega^2 reparameterization (task 1): does the sqrt-curvature of a
    # linear-in-omega^2 response explain the flags above? ---
    print(f"\n{material}: omega^2 reparameterization")
    flagged_keys_off = {(iq, br) for iq, br, _, _ in fnd_off + fd_off}
    sqrt_model_check(offgrid_rows, flagged_keys_off)
    binned_flag_rate(offgrid_rows, flagged_keys_off)

    gxy2_bz = np.array([r["gamma_omega2"] for r in offgrid_rows
                         if not r["unstable"] and not r["acoustic"] and np.isfinite(r["gamma_omega2"])])
    rms_gxy2 = float(np.sqrt(np.mean(gxy2_bz ** 2)))
    mean_gxy2 = float(np.mean(gxy2_bz))
    pin_ratio2 = abs(mean_gxy2) / rms_gxy2 if rms_gxy2 > 0 else float("nan")
    print(f"{material}: [omega^2] BZ-weighted <gamma_omega2>={mean_gxy2:.6f}  "
          f"rms(gamma_omega2)={rms_gxy2:.6f}  |<gamma_omega2>|/rms={pin_ratio2:.6f}  "
          f"[{'FAIL' if pin_ratio2 > 0.01 else 'PASS'} vs 0.01 threshold]")

    fnd_off2, fd_off2, n_nd_off2, n_d_off2 = report_flags_omega2(offgrid_rows, rms_gxy2, "omega^2, off-grid 11x11x11")
    ongrid_flag_rate2 = float("nan")
    if ongrid_rows is not None:
        fnd_on2, fd_on2, n_nd_on2, n_d_on2 = report_flags_omega2(ongrid_rows, rms_gxy2, "omega^2, on-grid 4x4x4")
        ongrid_flag_rate2 = len(fnd_on2) / n_nd_on2 if n_nd_on2 else float("nan")

    delta_plus2, delta_minus2 = eta_sensitivity_omega2(offgrid_rows)
    worst_eta2 = max(abs(delta_plus2), abs(delta_minus2))
    print(f"{material}: [omega^2] eta_44 sensitivity: Delta_eta/eta = "
          f"{delta_plus2:+.4f} (+) / {delta_minus2:+.4f} (-)  worst={worst_eta2:.4f}  "
          f"[{'FAIL' if worst_eta2 >= ETA_SENSITIVITY_FAIL else 'PASS'} vs {ETA_SENSITIVITY_FAIL} threshold]")

    decision_pass_omega2 = (
        ongrid_rows is not None
        and np.isfinite(ongrid_flag_rate2)
        and ongrid_flag_rate2 < 0.05
        and worst_eta2 < 0.05
    )
    print(f"{material}: DECISION (omega^2 basis, task 2 criteria: on-grid flags < 5% "
          f"AND worst |Delta_eta/eta| < 5%) — on-grid flag rate = "
          f"{100.0 * ongrid_flag_rate2:.1f}%, worst |Delta_eta/eta| = {worst_eta2:.4f} "
          f"-> {'PASS: eps05 adequate on omega^2 basis' if decision_pass_omega2 else 'FAIL: material curvature persists in omega^2 -- hold production gamma'}")

    out = REPO / "data" / "processed" / "reports" / f"nonlinearity_{material}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# nonlinearity_{material}.csv - produced by scripts/check_shear_nonlinearity.py",
        "# (A5' scope cut, 2026-07-18; q-convention + degenerate-pairing fixed 2026-07-20,",
        "# see G1' incident log in data/processed/reports/GATE_1p.md)",
        f"# BZ-weighted <gamma_xy>={mean_gxy:.6f} rms(gamma_xy)={rms_gxy:.6f} "
        f"|<gamma_xy>|/rms={pin_ratio:.6f} ({'FAIL' if pin_ratio > 0.01 else 'pass'})",
        f"# off-grid 11^3: non-degenerate {n_nd_off} stable, {len(fnd_off)} flagged "
        f"({100.0 * len(fnd_off) / (n_nd_off or 1):.1f}%); degenerate {n_d_off} stable, "
        f"{len(fd_off)} flagged ({100.0 * len(fd_off) / (n_d_off or 1):.1f}%)",
    ]
    if ongrid_rows is not None:
        header.append(
            f"# on-grid 4^3 (exact): non-degenerate {n_nd_on} stable, {len(fnd_on)} flagged "
            f"({100.0 * len(fnd_on) / (n_nd_on or 1):.1f}%); degenerate {n_d_on} stable, "
            f"{len(fd_on)} flagged ({100.0 * len(fd_on) / (n_d_on or 1):.1f}%)"
        )
    header.append(
        f"# eta_44 sensitivity @ {TEMPERATURE_K:.0f}K: Delta_eta/eta = {delta_plus:+.4f}(+)/"
        f"{delta_minus:+.4f}(-), worst={worst_eta_sensitivity:.4f}"
    )
    header.append(f"# DECISION (omega-space): {'PASS' if decision_pass_omega else 'FAIL'}")
    header.append(
        f"# omega^2 basis: BZ <gamma_omega2>={mean_gxy2:.6f} rms={rms_gxy2:.6f} "
        f"|mean|/rms={pin_ratio2:.6f}; on-grid non-degenerate flag rate="
        f"{100.0 * ongrid_flag_rate2:.1f}%; eta_44 sensitivity worst={worst_eta2:.4f}"
    )
    header.append(f"# DECISION (omega^2-space, task 2 criteria): {'PASS' if decision_pass_omega2 else 'FAIL'}")
    header.append("q_index,qx,qy,qz,branch,omega_ref_cm1,gamma_xy,a1_linear,a2_curvature,"
                   "curvature_over_rms_gamma,D_domega2,b2_curvature_omega2,gamma_omega2,"
                   "a2_pred_sqrt,unstable_flag,degenerate_flag,flag_gt_threshold")
    rows_out = []
    for r in offgrid_rows:
        curvature_gamma = abs(r["a2"] * H / r["omega_ref"])
        ratio = curvature_gamma / rms_gxy if rms_gxy > 0 else float("nan")
        flag = (not r["unstable"]) and np.isfinite(ratio) and ratio > FLAG_FRACTION
        q = r["q"]
        rows_out.append(
            f"{r['iq']},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{r['branch']},"
            f"{r['omega_ref']:.4f},{r['gamma_xy']:.4f},{r['a1']:.4f},{r['a2']:.4f},"
            f"{ratio:.4f},{r['D']:.4f},{r['b2']:.4f},{r['gamma_omega2']:.4f},"
            f"{r['a2_pred_sqrt']:.4f},{int(r['unstable'])},{int(r['degenerate'])},{int(flag)}"
        )
    out.write_text("\n".join(header) + "\n" + "\n".join(rows_out) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    for material in sys.argv[1:] or ["SrTiO3", "BaTiO3"]:
        process(material)
