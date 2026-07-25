#!/usr/bin/env python3
"""Stage C: Tamura 18O isotope series eta(f) for SrTiO3 at 300 K, plus the
kinetic-theory and Akhiezer external anchors.

Isotope channel: Tamura mass-variance scattering on the oxygen sublattice,

    1/tau_iso(omega) = (pi/6) V0 g2 omega^2 g(omega),

with g2 from the EXACT site-resolved sum (latvisc.isotope.mass_variance_g2 —
verified against manuscript Eq. (9), see isotope_g2_verification.csv), V0 the
volume per atom, and g(omega) the total phonon DOS per volume per rad/s built
from the renormalized frequencies of this same pipeline (Gaussian smearing).
Simplification, stated: the |e_kappa|^2 site projection of Eq. (10) is
approximated by the TOTAL DOS rather than the O-partial DOS; since the O
sublattice dominates the DOS above ~150 cm-1 and the isotope rate rises as
omega^2, this overestimates the low-frequency isotope scattering somewhat —
the reported eta suppression is therefore an upper bound at fixed g2.
Linewidths combine by Matthiessen: Gamma_total = Gamma_anh + Gamma_iso
(HWHM convention, Gamma_iso = (1/tau_iso)/2).

Anchors (logged as computed, no cherry-picking):
  * kinetic-theory estimate eta_kin = 3 n_at k_B T <gamma^2 tau> using the
    same per-mode gamma/tau as the full sum (plain mode average) — agreement
    within a factor of a few is the manuscript's own consistency requirement;
  * Akhiezer conversions at 1 GHz with MEASURED sound velocity and density
    (Bell & Rupprecht 1963 via sound_velocities.csv conventions):
    alpha = omega^2 eta/(2 rho v^3), Q^-1 = omega eta/(rho v^2).

Reads : same inputs as compute_eta_SrTiO3.py
Writes: data/processed/eta_isotope_SrTiO3.csv

Usage: uv run python scripts/compute_eta_isotope_SrTiO3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
from compute_eta_SrTiO3 import (  # noqa: E402
    CM1, N_Q, OMEGA_MIN, REPO, V_CELL, assemble, build_maps, load_vogt,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.isotope import isotope_scattering_rate, mass_variance_g2  # noqa: E402
from latvisc.validation import (  # noqa: E402
    akhiezer_attenuation, inverse_quality_factor, kinetic_viscosity_estimate,
)

M_NAT_O = 15.999
M_18O = 17.99916
FRACTIONS = [0.0, 0.01, 0.05, 0.10, 0.15]
T_K = 300
V0 = V_CELL / 5.0                       # volume per atom
DOS_SIGMA_CM1 = 10.0
RHO_MEAS = 5110.0                       # kg/m^3, Bell & Rupprecht 1963
V_SHEAR_MEAS = 4900.0                   # m/s, ~sqrt(c44/rho) transverse, measured scale
F_ACOUSTIC_HZ = 1.0e9


def build_dos(rows):
    """Total DOS per (m^3 rad/s) from mapped renormalized 300 K frequencies."""
    map_lambda = build_maps(T_K)[0]
    omegas = []
    for r in rows:
        if r["acoustic"]:
            continue
        lam_r = map_lambda(np.sign(r["omega_ref"]) * r["omega_ref"] ** 2)
        if lam_r > 0:
            omegas.append(np.sqrt(lam_r))
    omegas_rad = np.asarray(omegas) * CM1
    sigma = DOS_SIGMA_CM1 * CM1

    def dos(omega_rad):
        x = (np.asarray(omega_rad)[..., None] - omegas_rad[None, ...]) / sigma
        return (np.exp(-0.5 * x * x).sum(axis=-1)
                / (sigma * np.sqrt(2.0 * np.pi) * V_CELL * N_Q))
    return dos


def main() -> None:
    rows = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    vogt, _ = load_vogt()
    dos = build_dos(rows)

    print(f"Tamura 18O series at {T_K} K (exact g2 sum; total-DOS approximation "
          f"for the site projection, see docstring)")
    out_rows = ["f_18O,g2,eta_total_Pas,eta_over_eta0"]
    eta0 = None
    for f in FRACTIONS:
        if f == 0.0:
            g2 = 0.0
            extra = None
        else:
            g2 = mass_variance_g2([M_NAT_O, M_18O], [1.0 - f, f])

            def extra(omega_r_cm1, g2=g2):
                w = omega_r_cm1 * CM1
                rate = isotope_scattering_rate(w, g2, V0, dos(w))
                return float(np.squeeze(rate)) / 2.0 / CM1   # HWHM, cm-1
        eta, sec, flags = assemble(T_K, rows, vogt, extra_gamma_hwhm_cm1=extra)
        if eta0 is None:
            eta0 = eta
        print(f"  f={f:4.2f}: g2={g2:.3e}  eta={eta:.4e} Pa s  "
              f"eta/eta(0)={eta / eta0:.4f}")
        out_rows.append(f"{f},{g2:.6e},{eta:.6e},{eta / eta0:.6f}")

    monotone = all(float(out_rows[i].split(",")[2]) >= float(out_rows[i + 1].split(",")[2])
                   for i in range(1, len(out_rows) - 1))
    print(f"eta decreases monotonically with f: "
          f"{'YES — corrected physics confirmed (eta ∝ tau)' if monotone else 'NO — INVESTIGATE'}")

    # ---- external anchors (logged as computed) ----
    print(f"\nExternal anchors at {T_K} K (eta_0 = {eta0:.4e} Pa s):")
    # same per-mode gamma/tau as the full sum, via assemble's audit details
    _, _, _, details = assemble(T_K, rows, vogt, return_details=True)
    gam_sq_tau = [d["gruneisen"] ** 2 * d["tau_s"] for d in details]
    n_at = 5.0 / V_CELL
    eta_kin = kinetic_viscosity_estimate(n_at, T_K, float(np.mean(gam_sq_tau)))
    print(f"  kinetic estimate eta_kin = 3 n_at kB T <gamma^2 tau> = {eta_kin:.4e} Pa s"
          f"  -> ratio eta_full/eta_kin = {eta0 / eta_kin:.2f} "
          f"(consistency requires O(1) within a factor of a few)")

    omega_ac = 2.0 * np.pi * F_ACOUSTIC_HZ
    alpha = akhiezer_attenuation(omega_ac, eta0, RHO_MEAS, V_SHEAR_MEAS)
    q_inv = inverse_quality_factor(omega_ac, eta0, RHO_MEAS, V_SHEAR_MEAS)
    print(f"  Akhiezer at 1 GHz (measured rho={RHO_MEAS:.0f}, v_s={V_SHEAR_MEAS:.0f}):"
          f" alpha = {float(alpha):.3e} 1/m = {float(alpha) * 8.686e-2:.3f} dB/cm,"
          f" Q^-1 = {float(q_inv):.3e}")
    print("  (experimental comparison values to be attached from the literature"
          " anchors — logged as computed, none omitted)")

    out = REPO / "data" / "processed" / "eta_isotope_SrTiO3.csv"
    header = [
        "# eta_isotope_SrTiO3.csv - produced by scripts/compute_eta_isotope_SrTiO3.py",
        f"# Tamura 18O series at {T_K} K, exact g2 sum (Eq. 9), total-DOS site-projection",
        "# approximation, Matthiessen Gamma_total = Gamma_anh + Gamma_iso (HWHM).",
        "# Baseline eta carries the same Stage-C provenance/provisional flags as",
        "# eta_SrTiO3.csv (see eta_SrTiO3_stageC.md), incl. the OUT-OF-DECADE gate.",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(out_rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
