#!/usr/bin/env python3
"""Audit 0.C (parts 1-2): which stress variable does the two-pole kernel describe?

Classical damped harmonic oscillator (mass 1), x'' + 2 Gamma x' + omega0^2 x = xi(t),
equilibrium white noise, <x^2> = k_B T / omega0^2, <v^2> = k_B T.

Two candidate stress variables, both Gaussian-factorised (Wick):
  (E)  the oscillator ENERGY  E = (v^2 + omega0^2 x^2)/2   -> a^dag a - diagonal
       (DeVault occupation stress); its normalised time integral is the
       production kernel tau_two_pole_exact = 1/(2 Gamma) + Gamma/(2 omega0^2).
  (S)  the STIFFNESS-CONJUGATE stress S = dH/d eps = (1/2)(d omega0^2/d eps) x^2
       = -gamma omega0^2 x^2; its time integral, normalised to gamma^2 (k_B T)^2
       (the classical limit of (hbar omega)^2 gamma^2 n(n+1)), is
       tau_two_pole_stress = 1/(2 Gamma) + 2 Gamma/omega0^2.
The difference is exactly the a a / a^dag a^dag content of x^2 that the
manuscript's approximation (ii) drops.

Checks performed here:
  1. Direct numerical integration (scipy.quad) of both correlators built from
     the regression-theorem solution of the oscillator ODE, at
     Gamma/omega0 = 0.1, 1, 2.7, 5, 8.5, 10 -> reproduces both closed forms.
  2. Frequency-domain (spectral-function) evaluation of the same two
     integrals with the two-pole spectral function Im chi(w) =
     2 Gamma w / ((omega0^2 - w^2)^2 + 4 Gamma^2 w^2) and the Bose factors
     kept INSIDE the integral (the route of manuscript Eq. A7):
        I_E = (hbar^2 / 2 pi) int dw Imchi(w)^2 (w^2 + omega0^2)^2 n(w)(n(w)+1)
        I_S = gamma^2 omega0^4 (2 hbar^2 / pi) int dw Imchi(w)^2 n(w)(n(w)+1)
     (integrals over all w; the integrand is even), compared with
     (hbar omega0)^2 n(n+1) tau_kernel for ~10 representative Route-H modes
     at 300 K spanning hbar omega/k_B T = 0.1-0.9 and Gamma/omega = 0.05-3.
     In the classical limit each ratio must return to 1 (cross-check of the
     time- and frequency-domain evaluations).
  3. The single-Lorentzian (a-operator) spectral function with Bose factors
     inside is shown to be ill-defined for any finite width: near w -> 0,
     n(n+1) ~ (k_B T / hbar w)^2 while the Lorentzian stays finite, so the
     integral diverges; only the two-pole (odd) spectral function, which
     vanishes linearly at w = 0, gives a finite integral.

Writes: data/processed/v3_diagnostics/kernel_classical_check.csv
        data/processed/v3_diagnostics/kernel_quantum_check.csv
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR
from scipy.integrate import quad, solve_ivp

from diag_common import CM1, REPO, DIAG_DIR, mode_table, tau_two_pole_exact, tau_two_pole_stress

RATIOS = [0.1, 1.0, 2.7, 5.0, 8.5, 10.0]
T_K = 300.0


# ---------------------------------------------------------------- classical
def regression_correlator(gamma, omega0=1.0):
    """C_xx(t)/<x^2> from the regression theorem: the normalised correlator
    obeys the oscillator ODE with c(0) = 1, c'(0) = 0 (<x v> = 0 in
    equilibrium). Returns c(t) (analytic, complex-safe) and its derivative."""
    if abs(omega0**2 - gamma**2) < 1e-9 * omega0**2:
        gamma = gamma * (1.0 + 1e-6)          # step off exact critical damping (removable singularity)
    if gamma < omega0:                        # underdamped: oscillatory form
        w1 = np.sqrt(omega0**2 - gamma**2)

        def c(t):
            return np.exp(-gamma * t) * (np.cos(w1 * t) + (gamma / w1) * np.sin(w1 * t))

        def cdot(t):
            return -(omega0**2 / w1) * np.exp(-gamma * t) * np.sin(w1 * t)
    else:                                     # overdamped: two real exponentials (overflow-safe)
        kappa = np.sqrt(gamma**2 - omega0**2)
        lam_slow, lam_fast = gamma - kappa, gamma + kappa

        def c(t):
            return 0.5 * ((1 + gamma / kappa) * np.exp(-lam_slow * t) + (1 - gamma / kappa) * np.exp(-lam_fast * t))

        def cdot(t):
            return -(omega0**2 / (2 * kappa)) * (np.exp(-lam_slow * t) - np.exp(-lam_fast * t))

    return c, cdot


def check_regression_against_ode(gamma, omega0=1.0):
    """Spot check of the analytic correlator against a direct ODE solve."""
    c, _ = regression_correlator(gamma, omega0)
    sol = solve_ivp(lambda t, y: [y[1], -2 * gamma * y[1] - omega0**2 * y[0]],
                    (0, 20 / min(gamma, omega0)), [1.0, 0.0], rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0, 20 / min(gamma, omega0), 400)
    return float(np.max(np.abs(sol.sol(ts)[0] - np.array([c(t) for t in ts]))))


def classical_integrals(gamma, omega0=1.0):
    """Time integrals (0, inf) of the two normalised correlators."""
    c, cdot = regression_correlator(gamma, omega0)
    # C_xx = <x^2> c ; C_vx = <x^2> cdot ; C_vv = -<x^2> cddot = <x^2>(2 gamma cdot + omega0^2 c)
    # <x^2> = kT/omega0^2 ; <v^2> = kT.
    # energy: <dE dE>/(kT)^2 = (1/2)[C_vv^2 + omega0^4 C_xx^2 + 2 omega0^2 C_vx^2]/(kT)^2
    def e_corr(t):
        cc, cd = c(t), cdot(t)
        cvv = (2 * gamma * cd + omega0**2 * cc) / omega0**2      # C_vv / kT
        cxx = cc                                              # omega0^2 C_xx / kT
        cvx = cd / omega0                                     # omega0 C_vx / kT
        return 0.5 * (cvv**2 + cxx**2 + 2 * cvx**2)

    # stress: <dS dS>/(gamma^2 (kT)^2) = 2 omega0^4 C_xx^2/(kT)^2 = 2 c^2
    def s_corr(t):
        return 2.0 * c(t) ** 2

    tmax = 60.0 / min(gamma, omega0**2 / gamma)
    ie = quad(e_corr, 0, tmax, limit=4000, points=[1.0 / max(gamma, omega0)])[0]
    is_ = quad(s_corr, 0, tmax, limit=4000, points=[1.0 / max(gamma, omega0)])[0]
    return ie, is_


# ------------------------------------------------------------------ quantum
def im_chi(w, omega0, gamma):
    return 2.0 * gamma * w / ((omega0**2 - w**2) ** 2 + 4.0 * gamma**2 * w**2)


def bose_nn1(w, temperature):
    """n(w)(n(w)+1), even in w, overflow-safe; -> (kT/hbar w)^2 for small w."""
    x = HBAR * np.abs(w) / (K_B * temperature)
    if x < 1e-6:
        return 1.0 / x**2
    if x > 700:
        return 0.0
    e = np.exp(x)
    return e / (e - 1.0) ** 2


def quantum_integrals(omega0, gamma, temperature, classical=False):
    """I_E/(hbar omega0)^2 and I_S/(gamma^2 (hbar omega0)^2), spectral route."""
    def nn1(w):
        if classical:
            return (K_B * temperature / (HBAR * w)) ** 2
        return bose_nn1(w, temperature)

    def f_e(w):
        return im_chi(w, omega0, gamma) ** 2 * (w**2 + omega0**2) ** 2 * nn1(w)

    def f_s(w):
        return im_chi(w, omega0, gamma) ** 2 * nn1(w)

    wmax = 40.0 * max(omega0, gamma)
    # segmented quadrature: the integrand is sharply peaked at omega0 +- gamma for
    # weakly damped modes, so the interval is split at omega0 +- k gamma.
    edges = sorted({0.0, wmax} | {max(0.0, min(wmax, omega0 + k * gamma))
                                  for k in (-64, -32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32, 64)})

    def seg(f):
        return sum(quad(f, a, b, limit=400)[0] for a, b in zip(edges[:-1], edges[1:]))

    ie = 2.0 * seg(f_e) / (2.0 * np.pi)                      # over all w, /2pi
    is_ = 2.0 * seg(f_s) * 2.0 / np.pi * omega0**4
    # I_E = (hbar^2/2pi) int (...); expressed per (hbar omega0)^2:
    return ie / omega0**2, is_ / omega0**2


def main() -> None:
    # ---------- 1. classical closed forms ----------
    print("Classical DHO (omega0 = 1): direct time integrals vs closed forms")
    print(f"{'Gamma/w0':>8} {'ODE err':>9} {'int E-corr':>11} {'tau_exact':>10} {'int S-corr':>11} "
          f"{'tau_stress':>11} {'S/E':>7}")
    rows = ["Gamma_over_omega,ode_spot_check_maxerr,int_energy_corr,tau_two_pole_exact,"
            "int_stress_corr,tau_two_pole_stress,ratio_stress_over_energy"]
    for r in RATIOS:
        err = check_regression_against_ode(r)
        ie, is_ = classical_integrals(r)
        te = float(tau_two_pole_exact(1.0, r))
        ts = float(tau_two_pole_stress(1.0, r))
        print(f"{r:8.2f} {err:9.1e} {ie:11.6f} {te:10.6f} {is_:11.6f} {ts:11.6f} {is_ / ie:7.3f}")
        assert abs(ie / te - 1) < 1e-6 and abs(is_ / ts - 1) < 1e-6, (r, ie, te, is_, ts)
        rows.append(f"{r},{err:.2e},{ie:.8f},{te:.8f},{is_:.8f},{ts:.8f},{is_ / ie:.6f}")
    out = DIAG_DIR / "kernel_classical_check.csv"
    out.write_text("# kernel_classical_check.csv - .C.1 (overdamped_kernel_check.py)\n"
                   "# Classical DHO, omega0 = 1, mass 1. int_energy_corr = int_0^inf <dE(t)dE(0)>dt/(kT)^2;\n"
                   "# int_stress_corr = int_0^inf <dS(t)dS(0)>dt/(gamma^2 (kT)^2) with S = -gamma omega0^2 x^2.\n"
                   "# Both reproduce their closed forms to 1e-6 (asserted).\n"
                   + "\n".join(rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")

    # ---------- 2. quantum spectral-route check on representative modes ----------
    eta, sec, flags, details = mode_table(300)
    targets_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    picked = []
    kt_cm1 = K_B * T_K / (HBAR * CM1)
    for x in targets_x:
        w_target = x * kt_cm1
        # nearest Route-H modes by omega_r; take the highest- and lowest-damping ones in a +-8% window
        cand = [d for d in details if abs(d["omega_r"] - w_target) < 0.08 * w_target and d["sector"] != "routeS"]
        if not cand:
            continue
        cand.sort(key=lambda d: d["gamma_hwhm"] / d["omega_r"])
        picked.append(cand[0])
        if len(cand) > 1:
            picked.append(cand[-1])
    # synthetic points to reach Gamma/omega = 3 (BaTiO3-like) at low hbar omega/kT
    synthetic = [(x * kt_cm1, r) for x in (0.1, 0.5, 0.9) for r in (0.05, 0.2, 0.5, 1.0, 3.0)]
    ratios = np.array([d["gamma_hwhm"] / d["omega_r"] for d in details])
    wts = np.array([d["eta_contrib"] for d in details]) / eta
    print(f"\nGamma/omega census at 300 K over {len(details)} modes: max {ratios.max():.4f}, "
          f"eta-weighted mean {np.sum(wts * ratios):.4f}, eta weight with Gamma/omega > 0.1: "
          f"{wts[ratios > 0.1].sum():.2e}, > 0.05: {wts[ratios > 0.05].sum():.2e}, > 0.02: {wts[ratios > 0.02].sum():.3f}")
    print(f"  eta-weighted <(Gamma/omega)^2> = {np.sum(wts * ratios**2):.2e} (relative size of the "
          f"Gamma/(2 omega^2) term in tau_two_pole_exact)")

    print("\nQuantum spectral-route integrals (Bose factors inside), 300 K, ratio to "
          "(hbar w)^2 n(n+1) x kernel")
    hdr = (f"{'omega_r':>8} {'Gamma':>7} {'G/w':>6} {'hw/kT':>6} {'I_E/wE':>8} {'I_S/wS':>8} "
           f"{'I_E/wS':>8} {'I_S/wE':>8} {'cl E':>7} {'cl S':>7}  weight  sector")
    print(hdr)
    qrows = ["source,iq,branch,sector,omega_r_cm1,Gamma_hwhm_cm1,Gamma_over_omega,hbar_omega_over_kT,"
             "eta_weight,I_E_over_w_tau_exact,I_S_over_w_tau_stress,I_E_over_w_tau_stress,"
             "I_S_over_w_tau_exact,classical_I_E_over_w_tau_exact,classical_I_S_over_w_tau_stress"]
    def do_one(label, iq, br, sector, w_cm1, g_cm1, weight):
        w = w_cm1 * CM1
        g = g_cm1 * CM1
        n = 1.0 / np.expm1(HBAR * w / (K_B * T_K))
        wgt = n * (n + 1.0)                                   # per (hbar w)^2
        te, ts = float(tau_two_pole_exact(w, g)), float(tau_two_pole_stress(w, g))
        ie, is_ = quantum_integrals(w, g, T_K)
        ie_c, is_c = quantum_integrals(w, g, T_K, classical=True)
        wgt_c = (K_B * T_K / (HBAR * w)) ** 2
        vals = (ie / (wgt * te), is_ / (wgt * ts), ie / (wgt * ts), is_ / (wgt * te),
                ie_c / (wgt_c * te), is_c / (wgt_c * ts))
        print(f"{w_cm1:8.1f} {g_cm1:7.2f} {g_cm1 / w_cm1:6.3f} {HBAR * w / (K_B * T_K):6.3f} "
              f"{vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f} {vals[4]:7.4f} {vals[5]:7.4f}  "
              f"{weight:.1e}  {sector} {label}")
        qrows.append(f"{label},{iq},{br},{sector},{w_cm1:.3f},{g_cm1:.4f},{g_cm1 / w_cm1:.4f},"
                     f"{HBAR * w / (K_B * T_K):.4f},{weight:.3e},"
                     + ",".join(f"{v:.6f}" for v in vals))

    for d in picked:
        do_one("mode", d["iq"], d["branch"], d["sector"], d["omega_r"], d["gamma_hwhm"],
               d["eta_contrib"] / eta)
    for w_cm1, ratio in synthetic:
        do_one("synthetic", -1, -1, "n/a", w_cm1, ratio * w_cm1, 0.0)

    out = DIAG_DIR / "kernel_quantum_check.csv"
    out.write_text("# kernel_quantum_check.csv - .C.2 (overdamped_kernel_check.py)\n"
                   "# Two-pole spectral function Im chi = 2 Gamma w/((w0^2-w^2)^2+4 Gamma^2 w^2), Bose factors\n"
                   "# inside the frequency integral (manuscript A7 route). I_E = energy (a^dag a) correlator,\n"
                   "# I_S = x^2-stress correlator; w = (hbar w0)^2 n(n+1) at the mode frequency; tau_exact =\n"
                   "# production kernel, tau_stress = 1/(2G)+2G/w0^2. classical_* columns replace n(n+1) by\n"
                   "# (kT/hbar w)^2 and must equal 1 (time-domain vs frequency-domain consistency).\n"
                   + "\n".join(qrows) + "\n")
    print(f"-> {out.relative_to(REPO)}")

    # ---------- 3. single Lorentzian with Bose inside: divergence demonstration ----------
    print("\nSingle-Lorentzian (a-operator) spectral function with n(n+1) inside, w0 = 100 cm-1, "
          "Gamma = 10 cm-1: partial integrals from w_min upward")
    w0, g = 100.0 * CM1, 10.0 * CM1

    def lor2_nn1(w):
        return (g / np.pi / ((w - w0) ** 2 + g**2)) ** 2 * bose_nn1(w, T_K)
    n0 = 1.0 / np.expm1(HBAR * w0 / (K_B * T_K))
    ref = n0 * (n0 + 1) / (2 * g)
    for wmin_frac in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        val = np.pi * quad(lor2_nn1, wmin_frac * w0, 40 * w0, points=[w0], limit=4000)[0]
        print(f"  w_min = {wmin_frac:.0e} w0: pi int L^2 n(n+1) / [n(n+1)/(2 Gamma)] = {val / ref:.4f}")
    print("  (grows without bound as w_min -> 0: the a-operator Lorentzian route with Bose factors\n"
          "   inside is ill-defined; the two-pole odd spectral function is finite.)")


if __name__ == "__main__":
    main()
