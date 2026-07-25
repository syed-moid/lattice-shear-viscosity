#!/usr/bin/env python3
"""Verify latvisc.isotope.mass_variance_g2 against manuscript Eq. (9) and
compute the planned isotope-series g2 values (18O on the O
sublattice, f = 0.05, 0.10, 0.15).

Eq. (9), Section2_Theory_v0.3_source.md Sec. 2.5:

    g2(kappa) = sum_i f_i(kappa) * (1 - M_i(kappa)/Mbar(kappa))^2

matches mass_variance_g2's implementation directly (same sum, same
normalization). For a 2-isotope mixture (natural-abundance O at fraction
1-f, pure 18O at fraction f) this reduces algebraically to the closed
form the manuscript quotes as a sanity number:

    g2 = f(1-f) * (Delta M / Mbar)^2,  Delta M = M_18O - M_nat_O

This script checks that closed form against the general sum numerically
(they must agree to float precision — it is an algebraic identity, not
an approximation) and reports g2(f) for the three specified fractions.

Does not compute Gamma^iso (Eq. 10) itself: that needs a real Gamma_anh(q,T)
baseline with q-resolved polarization vectors and a delta-function DOS sum,
which for BaTiO3 only exists at Gamma (literature anchors) and for SrTiO3
via the external ALAMODE dataset — not yet wired into this diagnostic.

Usage: uv run python scripts/verify_isotope_g2.py
Writes: data/processed/reports/isotope_g2_verification.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.isotope import mass_variance_g2  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Standard atomic weight (natural-abundance mixture) vs pure 18O.
M_NAT_O = 15.999
M_18O = 17.99916
FRACTIONS = [0.05, 0.10, 0.15]


def main() -> None:
    delta_m = M_18O - M_NAT_O
    print(f"M_nat_O = {M_NAT_O}, M_18O = {M_18O}, Delta M = {delta_m:.5f}, "
          f"Delta M / Mbar_nat = {delta_m / M_NAT_O:.5f}")

    # Exact algebraic identity for a 2-isotope mixture: g2 = f(1-f)(Delta
    # M / Mbar(f))^2 with the ACTUAL f-dependent mixture mean Mbar(f) =
    # M_nat + f*Delta M (not the fixed natural-abundance Mbar). This must
    # match the general sum to float precision -- it's a re-derivation of
    # the same formula, not an approximation.
    rows = ["f,g2_general_sum,g2_exact_closed_form,abs_diff,g2_manuscript_approx,manuscript_rel_error_pct"]
    print(f"\n{'f':>6}  {'g2 (general sum, Eq 9)':>24}  {'g2 (exact closed form)':>24}  {'|diff|':>10}")
    max_diff = 0.0
    for f in FRACTIONS:
        g2_general = mass_variance_g2([M_NAT_O, M_18O], [1.0 - f, f])
        mbar_f = M_NAT_O + f * delta_m
        g2_exact_closed = f * (1.0 - f) * (delta_m / mbar_f) ** 2
        diff = abs(g2_general - g2_exact_closed)
        max_diff = max(max_diff, diff)
        g2_manuscript_approx = f * (1.0 - f) * (delta_m / M_NAT_O) ** 2  # uses fixed Mbar_nat, per Sec 2.5's "~="
        rel_err_pct = 100.0 * (g2_manuscript_approx - g2_general) / g2_general
        print(f"{f:6.2f}  {g2_general:24.6e}  {g2_exact_closed:24.6e}  {diff:10.2e}")
        rows.append(f"{f},{g2_general:.8e},{g2_exact_closed:.8e},{diff:.2e},"
                    f"{g2_manuscript_approx:.8e},{rel_err_pct:.2f}")

    verdict = "PASS (identity holds to float precision)" if max_diff < 1e-12 else "FAIL"
    print(f"\nmass_variance_g2 vs exact 2-isotope closed form (Mbar(f), not fixed): {verdict} "
          f"(max |diff| = {max_diff:.2e})")

    print(f"\nmanuscript Sec. 2.5's simplified formula uses a FIXED Mbar_nat "
          f"(marked '~=' in the text, not an exact identity) -- this introduces "
          f"a real few-percent relative error at f=0.15 that grows with f (Mbar "
          f"shifts by f*Delta M/Mbar_nat = {FRACTIONS[-1] * delta_m / M_NAT_O * 100:.1f}% "
          f"at f=0.15). See per-f relative error column in the output CSV.")

    manuscript_coeff = 1.5e-2
    print(f"\nmanuscript's quoted coefficient ~1.5e-2 vs computed "
          f"(Delta M/Mbar_nat)^2 = {(delta_m / M_NAT_O) ** 2:.4e}: "
          f"{'consistent' if abs((delta_m / M_NAT_O) ** 2 - manuscript_coeff) < 2e-3 else 'MISMATCH'}")
    g2_exact_values = [mass_variance_g2([M_NAT_O, M_18O], [1.0 - f, f]) for f in FRACTIONS]
    print("g2 <= 2e-3 for f <= 0.15 (manuscript claim), using the EXACT g2 (general sum): "
          f"{'confirmed' if max(g2_exact_values) <= 2e-3 else 'NOT confirmed'} "
          f"(max exact g2 = {max(g2_exact_values):.4e})")

    print("\nNOTE: this verifies g2 (Eq. 9) only. Gamma^iso (Eq. 10) needs a "
          "q-resolved Gamma_anh baseline with polarization vectors and a "
          "delta-function DOS sum -- not available for BaTiO3 beyond "
          "Gamma-point literature anchors, and not yet wired to the "
          "SrTiO3 ALAMODE dataset. That remains a separate, larger pending item.")

    out = REPO / "data" / "processed" / "reports" / "isotope_g2_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# isotope_g2_verification.csv - produced by scripts/verify_isotope_g2.py",
        "# g2(kappa) = sum_i f_i (1 - M_i/Mbar)^2 (Eq. 9), 2-isotope O sublattice case",
        f"# M_nat_O={M_NAT_O}, M_18O={M_18O}, verdict={verdict}",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(rows) + "\n")
    print(f"\n-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
