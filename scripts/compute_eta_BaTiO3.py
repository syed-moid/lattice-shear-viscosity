#!/usr/bin/env python3
"""Stage C (BaTiO3): zone-center-anchored Route-H viscosity assembly.

SCOPE (Gamma-point-only, section 3.2 of the manuscript): after the fc3
cost-gate NO-GO, BaTiO3 has no full-zone Gamma_qs(T). What CAN be
computed with stated provenance is the soft-TO-branch contribution to
eta_44 anchored entirely at the zone center:

  eta_44^soft(T) = (1/k_B T) sum_b INT d^3q/(2pi)^3 (hbar w_q)^2
                   gamma_b(q)^2 n(n+1) tau_eff(w_q, Gamma_s)

  gamma_b(q) = Lambda_b / (2 w_q^2)      [Route H, Eq. (11)]
  w_q^2      = omega_s^2(T) + A_par q_par^2 + A_perp q_perp^2
  tau_eff    = overdamped-safe Eq. (12)  [latvisc.viscosity.tau_effective]

Inputs, all with per-row provenance:
  * Lambda_b: OWN strained-cell couplings of the two soft-TO components
    at Gamma (D of the bare-imaginary doublet, +/-1.17e5 cm-2/strain,
    symmetric split as E-symmetry requires — internal check);
  * omega_s(T), Gamma_s(T): MEASURED zone-center hyper-Raman series
    (VSR 1982 this-work, PRIMARY; softmode_inputs_BaTiO3.csv), linearly
    interpolated between measured points — NO Cochran/Curie-Weiss fit is
    imposed (VSR observe systematic deviation from the linear law);
    Gamma is HWHM (full damping already halved at build);
  * soft-branch dispersion: Harada 1971 neutron, A_par = 972 meV^2 A^2
    along the soft [100] axis (checked: predicts 79 cm-1 at
    q = 0.313 A^-1 / 423 K = Harada's own point), A_perp = 4750 (stiff);
  * integration capped at the measured-dispersion validity q_par <=
    0.47 A^-1 (Harada's range); cap sensitivity reported.

Stated approximations, with bias directions:
  * Gamma(q) = Gamma_s(T) (zone-center value everywhere): Harada show
    damping DECREASES with q, and in the overdamped regime tau_eff grows
    with Gamma — net direction not sign-definite, bounded by the cap
    sensitivity;
  * single-damped-oscillator parameterization of a response that is in
    reality two overlapping components (Presting Fig. 3, qualitatively;
    Hlinka 2008) — stated, not correctable at this scope;
  * the stable low-frequency manifold away from the soft branch (in
    SrTiO3: ~90% of eta) is NOT included — this is a PARTIAL, sector
    viscosity, lower bound with respect to the full-zone sum.

PRE-REGISTERED TEST (revised gate, GATE_1p 2026-07-24): full-zone
eta(300K-ish) expected in 1e-3..1e-2 Pa s. The zone-center sector alone
CANNOT reach that band if BaTiO3 resembles SrTiO3 (where the equivalent
sector carries ~3.5%); the test therefore reports (i) the sector value
unadjusted, (ii) the STO-sector-share-scaled inference, clearly labeled
as an inference. No inputs are tuned either way.

Reads : data/processed/softmode_inputs_BaTiO3.csv,
        data/raw/gruneisen_modes/BaTiO3/*.modes (Lambda extraction)
Writes: data/processed/eta_BaTiO3.csv

Usage: uv run python scripts/compute_eta_BaTiO3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.viscosity import bose_einstein, tau_effective  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CM1 = 2.0 * np.pi * speed_of_light * 100.0     # rad/s per cm-1
MEV_TO_CM1 = 8.065544
A_PAR = 972.0 * MEV_TO_CM1**2                   # cm-2 A^2, soft [100] axis
A_PERP = 4750.0 * MEV_TO_CM1**2                 # cm-2 A^2, stiff axes
Q_CAP_PAR = 0.47                                # A^-1, Harada validity range
OMEGA_MIN = 5.0
TEMPS = [410, 425, 450, 475, 500, 550, 600, 650, 700]
# measured conversions (Li, Chan, Grimsditch, Zouboulis 1991; cubic ~443 K)
RHO_MEAS = 5990.0
V_TA_MEAS = 4246.0


def load_zone_center():
    path = REPO / "data" / "processed" / "softmode_inputs_BaTiO3.csv"
    om, ga = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("quantity"):
            continue
        parts = line.split(",")
        if parts[4] == "vsr1982_fig3_thiswork":
            if parts[0] == "omega_s":
                om.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "Gamma_HWHM":
                ga.append((float(parts[1]), float(parts[2])))
    om.sort(); ga.sort()
    om_T = np.array([p[0] for p in om]); om_v = np.array([p[1] for p in om])
    ga_T = np.array([p[0] for p in ga]); ga_v = np.array([p[1] for p in ga])
    rng = (max(om_T.min(), ga_T.min()), min(om_T.max(), ga_T.max()))
    return (lambda T: float(np.interp(T, om_T, om_v)),
            lambda T: float(np.interp(T, ga_T, ga_v)), rng)


def soft_lambdas():
    """Strained-cell shear couplings of the Gamma soft-TO components."""
    rows = compute_dataset(MODES_DIR / "BaTiO3", MASSES["BaTiO3"], mesh_n=11)
    lams = [r["D"] for r in rows
            if r["iq"] == 0 and not r["acoustic"] and r["omega_ref"] < OMEGA_MIN]
    return np.array(lams)


def eta_sector(T, omega_s, gamma_hwhm, lambdas, u_cap):
    """Numerical radial integral in the scaled variable u (cm-1)."""
    u = np.linspace(1e-3, u_cap, 4000)
    omega = np.sqrt(omega_s**2 + u**2)                    # cm-1
    gam_sq_sum = float(np.sum(lambdas**2)) / (4.0 * omega**4)
    w = omega * CM1
    lw = gamma_hwhm * CM1
    occupation = bose_einstein(w, T)
    tau = tau_effective(w, lw)
    integrand = u**2 * (HBAR * w) ** 2 * gam_sq_sum * occupation * (occupation + 1.0) * tau
    radial = np.trapezoid(integrand, u)                   # J^2 s cm-3
    # d^3q [A^-3] = d^3u / sqrt(A_par A_perp^2); 1 A^-3 = 1e30 m^-3
    pref = 1e30 / (2.0 * np.pi**2 * np.sqrt(A_PAR) * A_PERP)
    return pref * radial / (K_B * T)


def main() -> None:
    omega_s, gamma_s, rng = load_zone_center()
    lambdas = soft_lambdas()
    print(f"soft-TO Lambda (strained cells, Gamma): {np.round(lambdas, 0)} cm-2/strain "
          f"(sum {np.sum(lambdas):+.0f} — E-symmetry split, must be ~0)")
    print(f"zone-center series (VSR this-work) covers T in [{rng[0]:.0f}, {rng[1]:.0f}] K")
    u_cap = np.sqrt(A_PAR) * Q_CAP_PAR

    out = ["T_K,eta_soft_sector_Pas,omega_s_cm1,Gamma_HWHM_cm1,overdamped,u_cap_cm1"]
    results = []
    for T in TEMPS:
        if not (rng[0] <= T <= rng[1]):
            continue
        om, ga = omega_s(T), gamma_s(T)
        eta = eta_sector(T, om, ga, lambdas, u_cap)
        eta_15 = eta_sector(T, om, ga, lambdas, 1.5 * u_cap)
        od = ga > om
        results.append((T, eta, om, ga, eta_15))
        print(f"T={T:3d} K: omega_s={om:5.1f}  Gamma_HWHM={ga:5.1f} cm-1 "
              f"({'overdamped' if od else 'underdamped'})  "
              f"eta_soft = {eta:.3e} Pa s  (cap x1.5: {eta_15:.3e}, "
              f"{100 * (eta_15 / eta - 1):+.0f}%)")
        out.append(f"{T},{eta:.6e},{om:.2f},{ga:.2f},{int(od)},{u_cap:.1f}")

    # pre-registered test, reported unadjusted
    etas = np.array([r[1] for r in results])
    t_arr = np.array([r[0] for r in results])
    i_ref = int(np.argmin(np.abs(t_arr - 410)))
    print(f"\nPRE-REGISTERED TEST vs the revised 1e-3..1e-2 Pa s expectation "
          f"(full-zone quantity):")
    print(f"  zone-center soft-sector value: {etas[i_ref]:.2e} (at {t_arr[i_ref]} K), "
          f"range {etas.min():.2e}..{etas.max():.2e} over {t_arr.min()}-{t_arr.max()} K")
    in_band = 1e-3 <= etas[i_ref] <= 1e-2
    print(f"  sector value in band: {'YES' if in_band else 'NO'} — but the sector is "
          f"PARTIAL by scope (in SrTiO3 the equivalent formerly-unstable+Gamma "
          f"sector carries ~3.5% of the full sum).")
    inferred = etas[i_ref] / 0.035
    print(f"  STO-sector-share-scaled INFERENCE (not a computation): full-zone "
          f"eta ~ {inferred:.1e} Pa s if BaTiO3 partitions like SrTiO3 "
          f"-> {'inside' if 1e-3 <= inferred <= 1e-2 else 'OUTSIDE'} the band.")
    print("  Verdict logged unadjusted; scope-qualified. Decision on how to report "
          "the test in section 4.5 rests with the author.")

    path = REPO / "data" / "processed" / "eta_BaTiO3.csv"
    header = [
        "# eta_BaTiO3.csv - produced by scripts/compute_eta_BaTiO3.py",
        "# ZONE-CENTER-ANCHORED soft-TO sector of eta_44 (Gamma-point-only scope,",
        "# manuscript 3.2): PARTIAL viscosity, lower bound wrt the full-zone sum.",
        "# Lambda from own strained cells; omega_s/Gamma from VSR 1982 hyper-Raman",
        "# (measured points, no Cochran fit); Harada dispersion, cap q_par<=0.47 A^-1.",
        "# See data/processed/reports/eta_BaTiO3_stageC.md for the full provenance.",
    ]
    path.write_text("\n".join(header) + "\n" + "\n".join(out) + "\n")
    print(f"-> {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
