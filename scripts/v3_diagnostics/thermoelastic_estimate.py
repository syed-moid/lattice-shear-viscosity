#!/usr/bin/env python3
"""classical thermoelastic attenuation of the LA wave at Maerten's
wavevector, for comparison with the measured 1-2 GHz damping band.

For a longitudinal wave the thermal-conduction (thermoelastic) loss in the
low-frequency regime D q^2 << Omega (Landau & Lifshitz, Fluid Mechanics
Sec. 79; Truell, Elbaum & Chick Ch. 3) is

    alpha_te = Omega^2 / (2 rho v_L^3) * kappa (1/c_v - 1/c_p),

with c_p - c_v = T beta^2 B_T / rho (per unit mass), beta the volumetric
thermal expansion and B_T the isothermal bulk modulus. The amplitude damping
rate is Gamma_te = alpha_te v_L. The regime check uses the thermal relaxation
time over a wavelength, tau_th = 1/(D q^2), D = kappa/(rho c_p): Omega tau_th
>> 1 means the loss is in the viscous-like (Omega^2) regime and the formula
above applies; in the opposite limit the loss saturates (Zener form
Delta_te Omega tau_th /(1 + Omega^2 tau_th^2), Delta_te = c_p/c_v - 1).

Inputs (SrTiO3, 300 K; see header of the CSV for sources):
    kappa = 12 W m^-1 K^-1, C_p = 98.9 J mol^-1 K^-1, M = 183.49 g mol^-1,
    rho = 5110 kg m^-3, linear alpha = 9.4e-6 K^-1 (beta = 3 alpha),
    B_T = (c11 + 2 c12)/3 with c11 = 316, c12 = 102.5 GPa, v_L = 7900 m/s.

Writes: data/processed/v3_diagnostics/thermoelastic_estimate_300K.csv
"""

from __future__ import annotations

import numpy as np

from diag_common import REPO, DIAG_DIR

T = 300.0
KAPPA = 12.0
CP_MOL = 98.9
M_MOL = 0.18349
RHO = 5110.0
ALPHA_LIN = 9.4e-6
C11, C12 = 316e9, 102.5e9
V_L = 7900.0
Q_LIST = [52e6, 55e6, 58e6]


def main() -> None:
    cp = CP_MOL / M_MOL                     # J/kg/K
    beta = 3 * ALPHA_LIN
    b_t = (C11 + 2 * C12) / 3
    cp_minus_cv = T * beta**2 * b_t / RHO
    cv = cp - cp_minus_cv
    eta_th = KAPPA * (1 / cv - 1 / cp)      # effective "thermal viscosity", Pa s
    D = KAPPA / (RHO * cp)
    delta = cp / cv - 1
    print(f"c_p = {cp:.1f} J/kg/K, c_p - c_v = {cp_minus_cv:.2f} J/kg/K (Delta_te = {delta:.4f}), "
          f"B_T = {b_t / 1e9:.1f} GPa, D = {D:.3e} m^2/s")
    print(f"effective thermal-conduction viscosity kappa (1/c_v - 1/c_p) = {eta_th:.3e} Pa s")
    rows = ["q_um_inv,f_GHz,Omega_tau_th,Gamma_te_GHz,Gamma_te_zener_GHz,eta_thermal_equiv_Pas"]
    for q in Q_LIST:
        om = V_L * q
        tau_th = 1 / (D * q**2)
        alpha_te = om**2 / (2 * RHO * V_L**3) * eta_th
        g_te = alpha_te * V_L
        # Zener/Debye form with the same relaxation strength (saturates when Omega tau_th < 1)
        q_inv_z = delta * (om * tau_th) / (1 + (om * tau_th) ** 2)
        g_z = 0.5 * q_inv_z * om        # Gamma_amp = omega Q^-1 / 2
        print(f"q = {q / 1e6:.0f} um-1 (f = {om / 2 / np.pi / 1e9:.1f} GHz): Omega tau_th = {om * tau_th:.1f}, "
              f"Gamma_te = {g_te / 1e9:.4f} GHz (Zener-form {g_z / 1e9:.4f} GHz) vs measured 1-2 GHz")
        rows.append(f"{q / 1e6:.0f},{om / 2 / np.pi / 1e9:.2f},{om * tau_th:.2f},{g_te / 1e9:.5f},{g_z / 1e9:.5f},{eta_th:.4e}")
    out = DIAG_DIR / "thermoelastic_estimate_300K.csv"
    out.write_text("# thermoelastic_estimate_300K.csv - (thermoelastic_estimate.py)\n"
                   "# Gamma_te = v_L Omega^2/(2 rho v_L^3) kappa(1/c_v - 1/c_p), Omega = v_L q; Zener column uses\n"
                   "# Delta_te Omega tau_th/(1+Omega^2 tau_th^2) with tau_th = 1/(D q^2), Gamma = omega Q^-1/2.\n"
                   "# Inputs: kappa = 12 W/m/K (300 K; e.g. Martelli et al. PRL 120, 125901 (2018) report ~11-12);\n"
                   "# C_p = 98.9 J/mol/K (Todd & Lorenson, JACS 74, 2043 (1952); NIST-JANAF ~99); M = 183.49 g/mol;\n"
                   "# rho = 5110 kg/m^3 and c11 = 316, c12 = 102.5 GPa (Bell & Rupprecht, Phys. Rev. 129, 90 (1963));\n"
                   "# linear thermal expansion 9.4e-6 /K at 300 K (de Ligny & Richet, PRB 53, 3013 (1996));\n"
                   "# v_L = 7900 m/s (Maerten et al. 7.9-8.1 nm/ps). Author to re-verify the cited values.\n"
                   + "\n".join(rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
