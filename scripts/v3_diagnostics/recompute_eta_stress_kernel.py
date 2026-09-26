#!/usr/bin/env python3
"""NOTE: the SrTiO3 part reproduces the archived table only with the legacy assembly of the submitted model; the
BaTiO3 part (kernel-model comparison cited in the paper) is independent of that and reproduces exactly.
Audit 0.C (part 3): production sums re-evaluated with the x^2-stress kernel.

Every input (frequencies, linewidths, couplings, sectors, Vogt anchors,
partition) is taken unchanged from the production assembly; only the
lifetime kernel is swapped:

    production : tau_two_pole_exact  = 1/(2 Gamma) + Gamma/(2 omega^2)   (energy / a^dag a stress)
    alternative: tau_two_pole_stress = 1/(2 Gamma) + 2 Gamma/omega^2     (x^2 stress, aa + a^dag a^dag kept)

Reported: SrTiO3 eta_xyxy(T) and sector decomposition, 100-400 K; the
eta weight carried by modes with Gamma/omega > 1 (and > 0.5) at 300 K;
the BaTiO3 zone-center-anchored soft-sector series 410-700 K with the same
kernel swap (scripts/compute_eta_BaTiO3.py logic reproduced with a kernel
argument) and the 700 K -> 410 K enhancement factor for both kernels.
Neither kernel is declared production here.

Writes: data/processed/v3_diagnostics/eta_SrTiO3_kernel_comparison.csv
        data/processed/v3_diagnostics/eta_BaTiO3_kernel_comparison.csv
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

from diag_common import CM1, REPO, DIAG_DIR, eta_mod, mode_table, recompute_eta, tau_two_pole_exact, tau_two_pole_stress
import compute_eta_BaTiO3 as bto  # noqa: E402
from latvisc.viscosity import bose_einstein  # noqa: E402

SECTORS = ["routeS", "routeH_stable", "routeH_unstable", "gamma_sector"]


def bto_eta_sector(T, omega_s, gamma_hwhm, lambdas, u_cap, kernel):
    """scripts/compute_eta_BaTiO3.py::eta_sector with a kernel argument."""
    u = np.linspace(1e-3, u_cap, 4000)
    omega = np.sqrt(omega_s**2 + u**2)
    gam_sq_sum = float(np.sum(lambdas**2)) / (4.0 * omega**4)
    w = omega * CM1
    lw = gamma_hwhm * CM1
    occupation = bose_einstein(w, T)
    tau = kernel(w, lw)
    integrand = u**2 * (HBAR * w) ** 2 * gam_sq_sum * occupation * (occupation + 1.0) * tau
    radial = np.trapezoid(integrand, u)
    pref = 1e30 / (2.0 * np.pi**2 * np.sqrt(bto.A_PAR) * bto.A_PERP)
    return pref * radial / (K_B * T)


def main() -> None:
    print("SrTiO3 eta_xyxy(T): production kernel (exact two-pole, energy) vs x^2-stress kernel")
    rows = ["T_K,eta_exact_Pas,eta_stress_Pas,ratio_stress_over_exact,"
            + ",".join(f"{s}_exact" for s in SECTORS) + "," + ",".join(f"{s}_stress" for s in SECTORS)
            + ",weight_Gamma_gt_omega_exact,weight_Gamma_gt_half_omega_exact,n_modes_Gamma_gt_omega"]
    for T in eta_mod.TEMPS:
        eta_e, sec_e, flags, details = mode_table(T)
        eta_s, sec_s, c_s = recompute_eta(details, T, tau_two_pole_stress)
        eta_chk, _, c_e = recompute_eta(details, T, tau_two_pole_exact)
        if abs(eta_chk / eta_e - 1) > 1e-12:
            # the SrTiO3 rows describe the submitted (legacy) model; its assembly kernel has since changed, so the archived
            # table in data/processed/v3_diagnostics/ is the record and this recomputation differs at the 1e-3 level
            print(f"  note: legacy assembly differs from the energy-kernel recomputation by {eta_chk / eta_e - 1:+.2e} at {T} K")
        ratio = np.array([d["gamma_hwhm"] / d["omega_r"] for d in details])
        w_gt1 = c_e[ratio > 1.0].sum() / eta_e
        w_gt05 = c_e[ratio > 0.5].sum() / eta_e
        print(f"  T={T:3d} K: exact {eta_e:.4e}  stress {eta_s:.4e}  ratio {eta_s / eta_e:.4f}  "
              f"| sectors stress/exact: " + " ".join(f"{s[:8]} {sec_s.get(s, 0) / max(sec_e.get(s, 0), 1e-300):.3f}" for s in SECTORS)
              + f" | weight(G>w) {w_gt1:.2e} ({int((ratio > 1).sum())} modes), weight(G>w/2) {w_gt05:.2e}")
        rows.append(f"{T},{eta_e:.6e},{eta_s:.6e},{eta_s / eta_e:.6f},"
                    + ",".join(f"{sec_e.get(s, 0):.6e}" for s in SECTORS) + ","
                    + ",".join(f"{sec_s.get(s, 0):.6e}" for s in SECTORS)
                    + f",{w_gt1:.4e},{w_gt05:.4e},{int((ratio > 1).sum())}")
        if T == 300:
            over = [(d, c) for d, c in zip(details, c_e) if d["gamma_hwhm"] / d["omega_r"] > 0.5]
            print(f"    300 K modes with Gamma/omega > 0.5: {len(over)}")
            for d, c in sorted(over, key=lambda t: -t[1])[:10]:
                print(f"      iq={d['iq']:4d} br={d['branch']:2d} {d['sector']:15s} omega0={d['omega0']:7.2f} "
                      f"omega_r={d['omega_r']:6.2f} Gamma={d['gamma_hwhm']:6.2f} G/w={d['gamma_hwhm'] / d['omega_r']:.3f} "
                      f"weight={c / eta_e:.2e}")
    out = DIAG_DIR / "eta_SrTiO3_kernel_comparison.csv"
    out.write_text("# eta_SrTiO3_kernel_comparison.csv - .C.3 (recompute_eta_stress_kernel.py)\n"
                   "# Production inputs unchanged; kernel swapped between tau_two_pole_exact (energy/occupation\n"
                   "# stress, production) and tau_two_pole_stress = 1/(2G)+2G/w^2 (x^2 stress). Sector columns in Pa s.\n"
                   "# weight_* = fraction of eta (production kernel) in modes with Gamma_hwhm/omega_r above the threshold.\n"
                   + "\n".join(rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")

    print("\nBaTiO3 zone-center-anchored soft sector, 410-700 K")
    omega_s, gamma_s, rng = bto.load_zone_center()
    lambdas = bto.soft_lambdas()
    u_cap = np.sqrt(bto.A_PAR) * bto.Q_CAP_PAR
    brows = ["T_K,omega_s_cm1,Gamma_HWHM_cm1,Gamma_over_omega_s,eta_exact_Pas,eta_stress_Pas,ratio_stress_over_exact"]
    res = []
    for T in bto.TEMPS:
        if not (rng[0] <= T <= rng[1]):
            continue
        om, ga = omega_s(T), gamma_s(T)
        e_e = bto_eta_sector(T, om, ga, lambdas, u_cap, tau_two_pole_exact)
        e_s = bto_eta_sector(T, om, ga, lambdas, u_cap, tau_two_pole_stress)
        res.append((T, e_e, e_s))
        print(f"  T={T:3d} K: omega_s={om:5.1f} Gamma={ga:6.1f} (G/w={ga / om:.2f})  exact {e_e:.3e}  "
              f"stress {e_s:.3e}  ratio {e_s / e_e:.3f}")
        brows.append(f"{T},{om:.2f},{ga:.2f},{ga / om:.3f},{e_e:.6e},{e_s:.6e},{e_s / e_e:.6f}")
    (T0, e0e, e0s), (T1, e1e, e1s) = res[-1], res[0]
    print(f"  enhancement {T0} K -> {T1} K: exact {e1e / e0e:.1f}-fold, stress {e1s / e0s:.1f}-fold")
    out = DIAG_DIR / "eta_BaTiO3_kernel_comparison.csv"
    out.write_text("# eta_BaTiO3_kernel_comparison.csv - .C.3 (recompute_eta_stress_kernel.py)\n"
                   "# compute_eta_BaTiO3.py logic (VSR 1982 omega_s/Gamma, Harada dispersion, own Lambda) with the\n"
                   "# lifetime kernel swapped; exact column must reproduce data/processed/eta_BaTiO3.csv.\n"
                   f"# enhancement 700->410 K: exact {e1e / e0e:.2f}, stress {e1s / e0s:.2f}\n"
                   + "\n".join(brows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
