#!/usr/bin/env python3
"""5-point omega^2(eps) analysis for the shear_xy strain series (A5' G1' follow-up).

Now that both strain magnitudes exist (+/-0.005 AND +/-0.010), this
validates the production eps05 central difference against a wider
baseline, using latvisc.gruneisen.match_four_strains_by_overlap so all
five points (reference + 4 strains) share one canonical per-branch
labeling instead of four independent 2-point matches (see the G1' cutoff
follow-up plan in data/processed/reports/GATE_1p.md).

Per (q, branch), on omega^2(eps) (the natural linear-response quantity,
see check_shear_nonlinearity.py's docstring for why not omega itself):

  D_eps05 = (omega2[+0.005] - omega2[-0.005]) / (2*0.005)   [inner pair]
  D_eps10 = (omega2[+0.010] - omega2[-0.010]) / (2*0.010)   [outer pair]
  D_richardson = (4*D_eps05 - D_eps10) / 3                  [Richardson
      extrapolation eliminating the leading O(h^2) finite-difference
      error term, i.e. the O(eps^3) piece of omega^2(eps) itself]
  D_cubic = c1 from a least-squares cubic fit omega^2(eps) = c0 + c1*eps
      + c2*eps^2 + c3*eps^3 over all 5 points (independent cross-check
      of D_richardson using all the data at once rather than a two-term
      combination)

gamma_xy is then -D/(2*omega_ref^2) for each estimator. Disagreement
between D_eps05 and D_richardson (relative to the BZ rms of gamma_xy, the
same well-posed metric used throughout this diagnostic family) measures
whether the existing production eps05-only estimate is already converged,
stratified by reference frequency omega0 -- this is exactly the check
flagged as pending in the G1' delta notes once the +/-1.0% pair landed.

Modes whose omega^2 changes sign somewhere across the sampled +/-0.010
strain range (a stability-boundary shift under strain) are excluded from
both the fit and the flag-rate denominator and counted separately.

Reads : data/raw/gruneisen_modes/<material>/{reference,shear_xy_m010,
        shear_xy_m005,shear_xy_p005,shear_xy_p010}.modes
Writes: data/processed/reports/richardson_5pt_<material>.csv

Usage: uv run python scripts/check_5point_richardson.py [SrTiO3] [BaTiO3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import match_four_strains_by_overlap  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODES_DIR = REPO / "data" / "raw" / "gruneisen_modes"

MASSES = {
    "SrTiO3": np.array([87.62, 47.867, 15.999, 15.999, 15.999]),
    "BaTiO3": np.array([137.327, 47.867, 15.999, 15.999, 15.999]),
}
DEGEN_TOL = 0.5
OMEGA_MIN = 5.0
ACOUSTIC_SKIP = 1e-3
FLAG_FRACTION = 0.10
OMEGA_BINS = [(0, 50), (50, 100), (100, 150), (150, 175), (175, 200), (200, 300), (300, np.inf)]
PROPOSED_CUTOFF = 175.0  # cm^-1, from the earlier omega^2-flag-rate scan (GATE_1p.md)

EPS = np.array([-0.010, -0.005, 0.0, 0.005, 0.010])
VANDER = np.vander(EPS, 4, increasing=True)          # (5,4): 1, eps, eps^2, eps^3
PINV_VANDER = np.linalg.pinv(VANDER)                  # (4,5)


def process(material: str) -> None:
    masses = MASSES[material]
    directory = MODES_DIR / material
    reference = read_modes(directory / "reference.modes")
    m010 = read_modes(directory / "shear_xy_m010.modes")
    m005 = read_modes(directory / "shear_xy_m005.modes")
    p005 = read_modes(directory / "shear_xy_p005.modes")
    p010 = read_modes(directory / "shear_xy_p010.modes")

    rows = []
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        _, freq_m010, vec_m010 = m010[iq]
        _, freq_m005, vec_m005 = m005[iq]
        _, freq_p005, vec_p005 = p005[iq]
        _, freq_p010, vec_p010 = p010[iq]

        matched_m010, matched_m005, matched_p005, matched_p010 = match_four_strains_by_overlap(
            freq_ref, vec_ref, freq_m010, vec_m010, freq_m005, vec_m005,
            freq_p005, vec_p005, freq_p010, vec_p010, masses, DEGEN_TOL,
        )

        # signed dynamical-matrix eigenvalue: matdyn prints imaginary
        # frequencies as NEGATIVE reals, so the eigenvalue is
        # sign(omega)*omega^2, NOT omega^2 (a bare square silently flips
        # the sign for any strained partner that has gone imaginary)
        def eigval(freq):
            return np.sign(freq) * freq ** 2

        omega2_ref = eigval(freq_ref)
        omega2 = np.stack([
            eigval(matched_m010), eigval(matched_m005), omega2_ref,
            eigval(matched_p005), eigval(matched_p010),
        ], axis=0)  # (5, nmodes)

        D_eps05 = (eigval(matched_p005) - eigval(matched_m005)) / (2.0 * 0.005)
        D_eps10 = (eigval(matched_p010) - eigval(matched_m010)) / (2.0 * 0.010)
        D_richardson = (4.0 * D_eps05 - D_eps10) / 3.0

        coeffs = PINV_VANDER @ omega2  # (4, nmodes): c0,c1,c2,c3
        D_cubic = coeffs[1]

        sign_change = (omega2.min(axis=0) <= 0.0) & (omega2_ref > 0.0)
        acoustic_q = np.linalg.norm(q) < ACOUSTIC_SKIP

        for branch in range(len(freq_ref)):
            unstable = bool(freq_ref[branch] < OMEGA_MIN)
            is_acoustic = bool(acoustic_q and abs(freq_ref[branch]) < OMEGA_MIN)
            rows.append({
                "iq": iq, "q": q, "branch": branch + 1,
                "omega_ref": float(freq_ref[branch]),
                "D_eps05": float(D_eps05[branch]),
                "D_eps10": float(D_eps10[branch]),
                "D_richardson": float(D_richardson[branch]),
                "D_cubic": float(D_cubic[branch]),
                "unstable": unstable, "acoustic": is_acoustic,
                "sign_change": bool(sign_change[branch]),
            })

    usable = [r for r in rows if not r["unstable"] and not r["acoustic"] and not r["sign_change"]]
    n_sign_change = sum(1 for r in rows if r["sign_change"])
    print(f"{material}: {len(rows)} (q,branch) entries; {len(usable)} usable "
          f"(excludes unstable, acoustic-Gamma, and {n_sign_change} sign-change-under-strain entries)")

    omega_ref_safe = np.array([r["omega_ref"] for r in usable])
    gamma_eps05 = -np.array([r["D_eps05"] for r in usable]) / (2.0 * omega_ref_safe ** 2)
    gamma_rich = -np.array([r["D_richardson"] for r in usable]) / (2.0 * omega_ref_safe ** 2)
    gamma_cubic = -np.array([r["D_cubic"] for r in usable]) / (2.0 * omega_ref_safe ** 2)
    rms_gamma = float(np.sqrt(np.mean(gamma_eps05 ** 2)))

    resid_rich = np.abs(gamma_eps05 - gamma_rich) / rms_gamma
    resid_cubic = np.abs(gamma_rich - gamma_cubic) / rms_gamma
    print(f"{material}: rms(gamma_xy, eps05) over usable modes = {rms_gamma:.6f}")
    print(f"{material}: internal cross-check |D_richardson - D_cubic|/rms max = "
          f"{float(np.max(resid_cubic)):.4f} (should be ~0; both use the same 5 points, "
          f"different combinations)")

    print(f"\n{material}: eps05-vs-Richardson agreement, stratified by omega0")
    header = "omega0_bin_cm1,n_usable,flag_rate_pct,median_resid,p90_resid"
    csv_rows = [header]
    for lo, hi in OMEGA_BINS:
        mask = (omega_ref_safe >= lo) & (omega_ref_safe < hi)
        n = int(mask.sum())
        label = f"[{lo:.0f},{hi if hi < np.inf else 'inf'})"
        if n == 0:
            print(f"  {label:>14} cm-1: n=0")
            csv_rows.append(f"{label},0,,,")
            continue
        r = resid_rich[mask]
        flagged = int(np.sum(r > FLAG_FRACTION))
        med, p90 = float(np.median(r)), float(np.percentile(r, 90))
        print(f"  {label:>14} cm-1: n={n:6d}  flagged(eps05 vs Richardson)={100.0 * flagged / n:5.1f}%  "
              f"median resid={med:.4f}  p90 resid={p90:.4f}")
        csv_rows.append(f"{label},{n},{100.0 * flagged / n:.2f},{med:.4f},{p90:.4f}")

    above = resid_rich[omega_ref_safe >= PROPOSED_CUTOFF]
    below = resid_rich[omega_ref_safe < PROPOSED_CUTOFF]
    flag_above = 100.0 * np.sum(above > FLAG_FRACTION) / len(above) if len(above) else float("nan")
    flag_below = 100.0 * np.sum(below > FLAG_FRACTION) / len(below) if len(below) else float("nan")
    verdict = "CONFIRMED" if flag_above < 100 * FLAG_FRACTION else "NOT CONFIRMED"
    print(f"\n{material}: Route-S/Route-H cutoff confirmation at omega0 = {PROPOSED_CUTOFF:.0f} cm-1 "
          f"(from the earlier omega^2-flag-rate scan): above-cutoff eps05-vs-Richardson flag rate = "
          f"{flag_above:.1f}% (n={len(above)}), below-cutoff = {flag_below:.1f}% (n={len(below)}) "
          f"-> {verdict} (eps05 is adequate above the cutoff)")

    out = REPO / "data" / "processed" / "reports" / f"richardson_5pt_{material}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# richardson_5pt_{material}.csv - produced by scripts/check_5point_richardson.py\n"
        f"# 5-point omega^2(eps) analysis, match_four_strains_by_overlap canonical labeling\n"
        f"# {n_sign_change} sign-change-under-strain entries excluded (see script docstring)\n"
        f"# proposed_cutoff_cm1={PROPOSED_CUTOFF:.0f}, flag_above_pct={flag_above:.2f}, "
        f"flag_below_pct={flag_below:.2f}, verdict={verdict}\n"
        + "\n".join(csv_rows) + "\n"
    )
    out.write_text(text)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    for material in sys.argv[1:] or ["SrTiO3"]:
        process(material)
