#!/usr/bin/env python3
"""budget_numbers.py - numbers on the v6 production model (hybrid model) (mode table v5 + the surface):
finite-frequency response, interbranch coherence exposure, numerical-derivative noise floor, kernel choice,
Gamma TO1 doublet Lambda_B/Lambda_A, and the conditional soft-branch taper sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
import compute_eta_SrTiO3 as em  # noqa: E402
from latvisc.coupling import diagonalise, project_coupling, strain_derivative_matrix, transfer_bare_coupling  # noqa: E402
from latvisc.viscosity import bose_einstein, tau_two_pole_exact, tau_two_pole_stress  # noqa: E402

t = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v6.csv", comment="#")
eta = t.eta_contrib_Pas.sum()
w = t.eta_contrib_Pas.values
tau = t.tau_stress_ps.values * 1e-12
print(f"eta(300 K) v5 = {eta:.4e} Pa s, {len(t)} modes")
# finite-frequency response eta'(Omega)/eta'(0) = <1/(1 + Omega^2 tau^2)>_eta
for f in (1e9, 70e9):
    om = 2 * np.pi * f
    print(f"  finite-frequency eta'({f / 1e9:.0f} GHz)/eta'(0) = {np.sum(w / (1 + (om * tau) ** 2)) / eta:.4f}; "
          f"eta-weighted <Omega tau> = {np.sum(w * om * tau) / eta:.3f}")
# interbranch coherence exposure: share of eta in modes with another mode at the same q within Gamma / 2 Gamma
exp1 = exp2 = 0.0
for iq, g in t.groupby("iq"):
    om = g.omega_r_cm1.values; gam = g.Gamma_hwhm_cm1.values; e = g.eta_contrib_Pas.values
    for i in range(len(g)):
        d = np.abs(om - om[i]); d[i] = np.inf
        if d.min() < gam[i]:
            exp1 += e[i]
        if d.min() < 2 * gam[i]:
            exp2 += e[i]
print(f"  coherence exposure (neighbour within Gamma / 2 Gamma at the same q): {100 * exp1 / eta:.1f} % / {100 * exp2 / eta:.1f} % of eta")
# noise floor: random delta Lambda added in quadrature
pref = w / t.Lambda_C_cm2.values ** 2
for dl in (100.0, 300.0, 1000.0):
    print(f"  noise floor delta Lambda = {dl:.0f} cm-2 (uncorrelated): eta bias +{100 * np.sum(pref * dl ** 2) / eta:.2f} %")
# kernel choice: energy-variable kernel instead of the stress kernel
CM1 = em.CM1
wr = t.omega_used_cm1.values * CM1; lw = t.Gamma_hwhm_cm1.values * CM1
tau_e = np.array([float(tau_two_pole_exact(a, b)) for a, b in zip(wr, lw)])
tau_s = np.array([float(tau_two_pole_stress(a, b)) for a, b in zip(wr, lw)])
print(f"  kernel: energy-variable / stress-correlator = {np.sum(w * tau_e / tau_s) / eta:.5f} (eta ratio); max mode Gamma/omega = {(t.Gamma_hwhm_cm1 / t.omega_used_cm1).max():.3f}")
# quantum kernel correction proxy: fraction of eta with Gamma/omega > 0.1 and > 0.3
r = (t.Gamma_hwhm_cm1 / t.omega_used_cm1).values
print(f"  eta share with Gamma/omega > 0.1: {100 * w[r > 0.1].sum() / eta:.2f} %, > 0.3: {100 * w[r > 0.3].sum() / eta:.2f} %")
# Gamma TO1 doublet on own od1: Lambda_B / Lambda_A
vogt, _ = em.load_vogt()
surf = em.load_construction_surface(300)
q = (0.0, 0.0, 0.0)
wb, Eb = diagonalise(em.qe_set("reference").dynmat(q))
wr0, Er = diagonalise(surf["sets"]["reference"].dynmat(q, asr_onsite=True))
K = strain_derivative_matrix(em.qe_set("shear_xy_p005").dynmat(q), em.qe_set("shear_xy_m005").dynmat(q), em.H_ENG)
Ksc = strain_derivative_matrix(surf["sets"]["shear_xy_p005"].dynmat(q, asr_onsite=True), surf["sets"]["shear_xy_m005"].dynmat(q, asr_onsite=True), em.H_ENG)
pb = project_coupling(K, Eb, wb); pB = project_coupling(K, Er, wr0); pC = project_coupling(Ksc, Er, wr0)
lamA, mu, ov1, ovm = transfer_bare_coupling(pb["lam"], Eb, Er, wr0)
to1 = [i for i in range(15) if 100 < wr0[i] < 160]
print(f"  Gamma TO1 triplet (own od1 omega_r {np.round(wr0[to1], 1)}): Lambda_A {np.round(lamA[to1], 0)}, Lambda_B {np.round(pB['lam'][to1], 0)}, "
      f"Lambda_C {np.round(pC['lam'][to1], 0)}; bare TO1 Lambda {np.round(pb['lam'][:3], 0)} (omega0 {np.round(wb[:3], 1)}); "
      f"overlap {np.round(ovm[to1], 2)}; sum B^2 / sum A^2 = {np.sum(pB['lam'][to1] ** 2) / np.sum(lamA[to1] ** 2):.3f}")
# conditional soft-branch taper sensitivity: omega_r^2 -> omega_r^2 - w(q) (omega_r(Gamma)^2 - omega_Vogt^2) on the TO1-partner
# subspace near Gamma (bare-imaginary partner, |q| <= q_taper), K and Gamma fixed, Bose/kernel recomputed
v_omega, v_gamma = vogt(300)
wG = float(np.mean(wr0[to1]))
d2 = wG ** 2 - v_omega ** 2
qv = t[["qa", "qb", "qc"]].values
qd = np.linalg.norm((qv + 0.5) % 1.0 - 0.5, axis=1)
T = 300.0
from scipy.constants import Boltzmann as K_B, hbar as HBAR  # noqa: E402
norm = 1.0 / (em.V_CELL * 1331 * K_B * T)
ratio2 = (v_omega / wG) ** 2                                    # Vogt^2 / own^2 at Gamma = 0.40
for label, cond in (("optical TO1 subspace (partner imaginary, omega_r >= 100)", (t.omega0_partner_cm1.values < 0) & (t.omega_r_cm1.values >= 100.0)),
                    ("all bare-imaginary-partner modes near Gamma", t.omega0_partner_cm1.values < 0)):
    for qt in (0.1, 0.2, 0.3):
        sub = cond & (qd <= qt) & (t.sector.values != "gamma_sector")
        wt = np.clip(1.0 - qd / qt, 0.0, 1.0)
        scale = 1.0 - wt * (1.0 - ratio2) * sub                        # omega^2 -> omega^2 * [1 - w(q)(1 - Vogt^2/own^2)]
        om2 = t.omega_r_cm1.values ** 2 * scale
        om_new = np.sqrt(om2)
        gam = -t.Lambda_C_cm2.values / (2 * om2)
        ws = om_new * CM1
        occ = bose_einstein(ws, T)
        tt = np.array([float(tau_two_pole_stress(a, b)) for a, b in zip(ws, lw)])
        contrib = (HBAR * ws) ** 2 * gam ** 2 * occ * (occ + 1) * tt * norm
        contrib = np.where(t.sector.values == "gamma_sector", w, contrib)
        print(f"  soft-branch taper [{label}] q_t = {qt}: {sub.sum()} modes, eta = {contrib.sum():.4e} ({100 * (contrib.sum() / eta - 1):+.1f} %); "
              f"min omega on the set {om_new[sub].min():.1f} cm-1 (was {t.omega_r_cm1.values[sub].min():.1f})")
