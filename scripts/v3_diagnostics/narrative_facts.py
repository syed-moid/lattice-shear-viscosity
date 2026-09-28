#!/usr/bin/env python3
"""narrative_facts_10.py - numbers from existing production data (mode table v6 (hybrid model), processed CSVs):
10.1 scale (kinetic anchor, O(1)-gamma reference with identical frequencies/linewidths/weights, sector shares, bins,
construction A analogue); 10.2 Maerten damping on production lifetimes; 10.4 isotope shares. Prints a block per item;
writes data/processed/v3_diagnostics/narrative_facts_10_bins.csv (eta by bare-omega_0 bin and by omega_r bin for C and A)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as K_B

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH, tee_stdout  # noqa: E402,F401  (sets sys.path)
tee_stdout("narrative_facts_300K.txt")
from compute_eta_SrTiO3 import V_CELL  # noqa: E402

T = 300.0
N_AT = 8.4e28          # m^-3 ; n_at = 5/a^3 = 8.47e28 with a = 3.8930 A
RHO, V_L = 5110.0, 8000.0
t = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v6.csv", comment="#")
eta = t.eta_contrib_Pas.values; E = eta.sum()
tau = t.tau_stress_ps.values * 1e-12; g2 = t.gamma_C.values ** 2
print(f"== 10.1 scale: eta_C(300 K) = {E:.4e} Pa s over {len(t)} modes; V_cell = {V_CELL:.4e} m^3; 15/V_cell = {15 / V_CELL:.3e} = 3 n_at with n_at = {5 / V_CELL:.3e}")
kin_a = 3 * N_AT * K_B * T * 2.0 * 1e-12
print(f"(a) O(1) benchmark 3 n_at k_B T <gamma^2> tau, n_at = {N_AT:.2e}, <gamma^2> = 2, tau = 1 ps: {kin_a:.3e} Pa s; full sum / (a) = {E / kin_a:.3f}")
wg2 = np.sum(eta * g2) / E; wtau = np.sum(eta * tau) / E
qw = t.n_n_plus_1.values * (t.omega_used_cm1.values * 1.986e-23 / (K_B * T)) ** 2   # (hbar w/kT)^2 n(n+1): classical limit -> 1
wq = np.sum(eta * qw) / E
plain = 3 * N_AT * K_B * T * np.mean(g2 * tau)          # plain mode average, classical weight
plain_q = 3 * N_AT * K_B * T * np.mean(g2 * tau * qw)   # plain mode average with the quantum weight
print(f"(b) eta-weighted <gamma_C^2> = {wg2:.3f}, <tau> = {wtau * 1e12:.3f} ps, <quantum weight (hbar w/kT)^2 n(n+1)> = {wq:.3f};")
print(f"    3 n_at k_B T <gamma^2 tau>_plain = {plain:.3e} (classical weight), {plain_q:.3e} (quantum weight); eta/plain_q = {E / plain_q:.3f}; eta/(3 n_at k_B T <g2>_eta <tau>_eta <w>_eta) = {E / (3 * N_AT * K_B * T * wg2 * wtau * wq):.3f}")
print(f"    zone-rms gamma_C = {np.sqrt(np.mean(g2)):.3f}; max |gamma_C| = {np.sqrt(g2.max()):.2f}; eta-weighted median |gamma_C| = {np.sqrt(g2[np.argsort(g2)][np.searchsorted(np.cumsum(eta[np.argsort(g2)]) / E, 0.5)]):.2f}")
# reference with identical frequencies, linewidths and weights, gamma^2 set to 1 and 2
ok = g2 > 0
for gref in (1.0, 2.0):
    ref = np.where(ok, eta * gref / np.where(ok, g2, 1.0), 0.0)
    print(f"    reference gamma^2 = {gref:.0f} (same omega, Gamma, weights): eta_ref = {ref.sum():.3e} Pa s; eta_C/eta_ref = {E / ref.sum():.3f}")
    for sec in ("imaginary", "[0,50)", "gamma_sector"):
        m = (t.sector.astype(str) == {"imaginary": "imaginary", "[0,50)": "[0-50)", "gamma_sector": "gamma_sector"}[sec]).values
        print(f"       sector {sec:12s}: eta_C {eta[m].sum():.3e} ({100 * eta[m].sum() / E:.1f} % of eta_C) vs ref {ref[m].sum():.3e} ({100 * ref[m].sum() / ref.sum():.1f} % of ref); ratio C/ref = {eta[m].sum() / ref[m].sum() if ref[m].sum() else np.nan:.3f}")
print("    sectors present:", sorted(t.sector.unique()))
print(f"    top-20 share C = {np.sort(eta)[-20:].sum() / E:.4f}")
# construction A analogue on the same surface
lamC, lamA = t.Lambda_C_cm2.values, t.Lambda_A_cm2.values
okA = lamC != 0
etaA = np.where(okA, eta * lamA ** 2 / np.where(okA, lamC ** 2, 1.0), 0.0); EA = etaA.sum()
gA2 = t.gamma_A.values ** 2
print(f"    construction A on the same surface: eta_A = {EA:.3e}; eta-weighted <gamma_A^2> = {np.sum(etaA * gA2) / EA:.2f}; top-20 share A = {np.sort(etaA)[-20:].sum() / EA:.3f}; A/C = {EA / E:.2f}")
# bins
def bins(x, edges):
    return pd.cut(x, edges, right=False)
edges0 = [-np.inf, 0, 25, 50, 100, 150, 175, 300, np.inf]; edgesr = [0, 25, 50, 100, 150, 175, 300, np.inf]
rows = []
for name, x, ed in (("omega0_partner", t.omega0_partner_cm1.values, edges0), ("omega_r", t.omega_r_cm1.values, edgesr)):
    b = bins(x, ed)
    for lab, idx in pd.Series(range(len(t))).groupby(b, observed=False):
        i = idx.values
        rows.append({"axis": name, "bin": str(lab), "n_modes": len(i), "eta_C_Pas": eta[i].sum(), "frac_C": eta[i].sum() / E,
                     "eta_A_Pas": etaA[i].sum(), "frac_A": etaA[i].sum() / EA,
                     "eta_wt_gamma_C2": np.sum(eta[i] * g2[i]) / eta[i].sum() if eta[i].sum() else np.nan,
                     "eta_wt_tau_ps": np.sum(eta[i] * tau[i]) / eta[i].sum() * 1e12 if eta[i].sum() else np.nan})
df = pd.DataFrame(rows); print(df.to_string(float_format=lambda v: f"{v:.4g}"))
with open(DIAG / "narrative_facts_v6_bins.csv", "w") as fh:
    fh.write("# (narrative_facts_10.py): eta by bare-omega_0 (partner) bin and by omega_r bin, constructions C (production) and A on the same surface, 300 K, 11^3.\n")
    df.to_csv(fh, index=False, float_format="%.6g")
# 10.2 Maerten
print("== 10.2 Maerten (production lifetimes, stress kernel, eta_xyxy as the shear proxy; rho = 5110, v_L = 8000 m/s)")
def eta_prime(om):
    return np.sum(eta / (1 + (om * tau) ** 2))
for q_um in (52.0, 55.0, 58.0):
    q = q_um * 1e6; om = V_L * q
    gs = E * q ** 2 / (2 * RHO); gd = eta_prime(om) * q ** 2 / (2 * RHO)
    print(f"    q = {q_um:.0f} um^-1 (f = {om / 2 / np.pi / 1e9:.1f} GHz): Gamma_static = {gs / 1e9:.3f} GHz, Gamma_dynamic = {gd / 1e9:.3f} GHz, eta'(Omega)/eta'(0) = {eta_prime(om) / E:.3f}")
qs = np.logspace(np.log10(0.4), 2, 60) * 1e6
gd = np.array([eta_prime(V_L * q) * q ** 2 / (2 * RHO) for q in qs])
ex = np.gradient(np.log(gd), np.log(qs))
for q_um in (0.4, 1, 3, 10, 30, 52, 58, 100):
    i = np.argmin(np.abs(qs / 1e6 - q_um)); print(f"    local exponent d ln Gamma_dyn / d ln q at {qs[i] / 1e6:.1f} um^-1: {ex[i]:.2f}")
# 10.4 isotope
print("== 10.4 isotope")
iso = pd.read_csv(REPO / "data" / "processed" / "eta_isotope_SrTiO3.csv", comment="#"); print(iso.to_string(index=False))
s0 = pd.read_csv(DIAG / "isotope_series_exact_projection.csv", comment="#")
print("    DOS-form series (submitted model, sigma 10):", s0[(s0.form == "dos") & (s0.sigma_cm1 == 10)][["f_18O", "eta_over_eta0"]].values.tolist())
if "Gamma_iso_f015_cm1" in t:
    gi = t.Gamma_iso_f015_cm1.values; ga = t.Gamma_hwhm_cm1.values; m = np.isfinite(gi)
    print(f"    eta-weighted Gamma_iso(f = 0.15) = {np.sum(eta[m] * gi[m]) / eta[m].sum():.4f} cm-1 vs eta-weighted Gamma_anh = {np.sum(eta[m] * ga[m]) / eta[m].sum():.4f} cm-1; ratio = {np.sum(eta[m] * gi[m]) / np.sum(eta[m] * ga[m]):.4f}; "
          f"eta-weighted <Gamma_iso/Gamma_anh> = {np.sum(eta[m] * gi[m] / ga[m]) / eta[m].sum():.4f}; modes with Gamma_iso available: {m.sum()}")
npz = np.load(REPO / "data" / "raw" / "alamode_sto" / "z_tut" / "i2s12" / "mesh11_z_reference_i2s12_od1_300K.npz")
om, ev = npz["omega_cm1"], npz["evec"]
iq, nu = t.iq.values.astype(int), t.nu.values.astype(int) - 1
chk = np.abs(om[iq, nu] - t.omega_r_cm1.values); print(f"    npz/mode-table frequency match: max |d omega| = {chk.max():.3f} cm-1 (Gamma triplet uses Vogt: excluded -> {np.sort(chk)[-4]:.3f})")
# evec[iq, mode, component]? test both layouts by normalisation
e1 = ev[iq, nu, :]; osh1 = np.sum(np.abs(e1[:, 6:15]) ** 2, axis=1) / np.sum(np.abs(e1) ** 2, axis=1)
e2 = ev[iq, :, nu]; osh2 = np.sum(np.abs(e2[:, 6:15]) ** 2, axis=1) / np.sum(np.abs(e2) ** 2, axis=1)
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402
al = AlamodeSet(REPO / "data" / "raw" / "alamode_sto" / "z_tut" / "i2s12" / "renorm_z_reference_i2s12_od1_300K.xml",
                REPO / "data" / "raw" / "alamode_sto" / "z_tut" / "z_reference.fc")
qf = npz["q_frac"][7]; wb, eb = diagonalise(al.dynmat(tuple(qf), asr_onsite=True)); o = np.argsort(wb); eb = eb[:, o]
ob = np.sum(np.abs(eb[6:15, :]) ** 2, axis=0) / np.sum(np.abs(eb) ** 2, axis=0)
o1 = np.sum(np.abs(ev[7][:, 6:15]) ** 2, axis=1) / np.sum(np.abs(ev[7]) ** 2, axis=1); o2 = np.sum(np.abs(ev[7][6:15, :]) ** 2, axis=0) / np.sum(np.abs(ev[7]) ** 2, axis=0)
print(f"    layout check at q index 7 {qf}: builder O-share per mode {np.round(ob, 3)}; npz [mode,comp] {np.round(o1, 3)}; npz [comp,mode] {np.round(o2, 3)}")
print(f"    eta-weighted O-site share (own eigenvectors): layout [iq,mode,comp] {np.sum(eta * osh1) / E:.3f}; layout [iq,comp,mode] {np.sum(eta * osh2) / E:.3f}; plain mean {osh1.mean():.3f}/{osh2.mean():.3f} (physical mean = 9/15 = 0.6)")
