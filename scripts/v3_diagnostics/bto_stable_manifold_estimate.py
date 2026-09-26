#!/usr/bin/env python3
"""order-of-magnitude estimate of the BaTiO3 stable-manifold
contribution missing from the zone-center-anchored soft sector (R3.3).

BaTiO3 has full-zone bare harmonic phonons and gamma_xy(qs) from its own
+-0.005 shear pair, but no Gamma_qs(T). For every stable bare mode (omega0
>= 5 cm-1, acoustic translations at Gamma excluded) the coupling is taken
Route-S-like, gamma = -D/(2 omega0^2), with bare-frequency Bose weights
(hbar omega0)^2 n(n+1) at T; the lifetime is assigned two ways:

  (a) SrTiO3 transfer: the median production tau_exact(300 K) of SrTiO3
      stable modes (Route S + Route H stable) in the 25 cm-1 bin of the
      renormalised frequency nearest to the BaTiO3 bare omega0, scaled by
      300 K / T;
  (b) Klemens-type Gamma = A <gamma^2> omega T, with A anchored on SrTiO3 at
      300 K (eta-weighted mean of Gamma/(<gamma^2>_STO omega 300 K) over the
      same stable set) and <gamma^2> the zone-mean gamma_xy^2 of each
      material's stable set; per-mode gamma_xy^2 is NOT used in Gamma because
      it vanishes on symmetry planes and would give unbounded lifetimes.

Both are estimates for the response letter, reported relative to the soft
sector at 410 K and 453 K; the bare-frequency denominators overestimate the
couplings of the soft-adjacent stable modes (renormalisation stiffens them),
so both numbers are biased high in that manifold. The Grueneisen convention
cancels in the ratio to the sector (both use the path derivative).

Reads : data/raw/gruneisen_modes/{SrTiO3,BaTiO3}/*.modes (via compute_dataset),
        data/processed/v3_diagnostics/mode_table_SrTiO3_300K.csv,
        data/processed/softmode_inputs_BaTiO3.csv (via compute_eta_BaTiO3)
Writes: data/processed/v3_diagnostics/bto_stable_manifold_estimate.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

from diag_common import CM1, REPO, DIAG_DIR, load_rows, tau_two_pole_exact
import compute_eta_BaTiO3 as bto  # noqa: E402
from recompute_eta_stress_kernel import bto_eta_sector  # noqa: E402

A_BTO = 4.0254e-10                  # m, PBE relaxed cell (materials.py); PBEsol strained cells use their own a
V_CELL_BTO = 65.22238e-30
N_Q = 11**3
TEMPS = [410.0, 453.0]
BIN = 25.0


def main() -> None:
    sto = pd.read_csv(DIAG_DIR / "mode_table_SrTiO3_300K.csv", comment="#")
    sto_stable = sto[sto["sector"].isin(["routeS", "routeH_stable"])]
    # (a) tau(omega) map from SrTiO3
    edges = np.arange(0, sto_stable["omega_r_cm1"].max() + BIN, BIN)
    centers, med_tau = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sto_stable["omega_r_cm1"] >= lo) & (sto_stable["omega_r_cm1"] < hi)
        if m.sum() >= 5:
            centers.append(0.5 * (lo + hi))
            med_tau.append(sto_stable.loc[m, "tau_exact_ps"].median() * 1e-12)
    centers, med_tau = np.array(centers), np.array(med_tau)
    # (b) Klemens anchor
    g2_sto = float(np.mean(sto_stable["gamma_xy_used"] ** 2))
    w_sto = sto_stable["eta_contrib_Pas"] / sto_stable["eta_contrib_Pas"].sum()
    A = float(np.sum(w_sto * sto_stable["Gamma_anh_cm1"] / (g2_sto * sto_stable["omega_r_cm1"] * 300.0)))
    print(f"SrTiO3 anchors: <gamma^2>_stable = {g2_sto:.3f}; Klemens A = {A:.3e} per K "
          f"(Gamma[cm-1] = A <gamma^2> omega[cm-1] T)")

    rows = load_rows("BaTiO3")
    stable = [r for r in rows if not r["acoustic"] and r["omega_ref"] >= 5.0]
    n_unstable = sum(1 for r in rows if not r["acoustic"] and r["omega_ref"] < 5.0)
    w0 = np.array([r["omega_ref"] for r in stable])
    D = np.array([r["D"] for r in stable])
    gam = -D / (2 * w0**2)
    g2_bto = float(np.mean(gam**2))
    print(f"BaTiO3: {len(stable)} stable mode-slots, {n_unstable} unstable (excluded, soft manifold); "
          f"<gamma^2>_stable = {g2_bto:.3f}, rms gamma = {np.sqrt(g2_bto):.3f}")
    omega_s, gamma_s, rng = bto.load_zone_center()
    lambdas = bto.soft_lambdas()
    u_cap = np.sqrt(bto.A_PAR) * bto.Q_CAP_PAR

    out_rows = ["T_K,eta_soft_sector_Pas,eta_stable_a_STO_tau_transfer_Pas,eta_stable_b_Klemens_Pas,"
                "ratio_a_over_sector,ratio_b_over_sector,frac_a_from_omega0_lt_100,frac_b_from_omega0_lt_100"]
    for T in TEMPS:
        w = w0 * CM1
        n = 1 / np.expm1(HBAR * w / (K_B * T))
        wt = (HBAR * w) ** 2 * n * (n + 1) * gam**2
        tau_a = np.interp(w0, centers, med_tau) * 300.0 / T
        gam_b_cm1 = A * g2_bto * w0 * T
        tau_b = tau_two_pole_exact(w, gam_b_cm1 * CM1)
        norm = 1 / (V_CELL_BTO * N_Q * K_B * T)
        eta_a = float(np.sum(wt * tau_a) * norm)
        eta_b = float(np.sum(wt * tau_b) * norm)
        sector = bto_eta_sector(T, omega_s(T), gamma_s(T), lambdas, u_cap, tau_two_pole_exact)
        low = w0 < 100
        fa = float(np.sum((wt * tau_a)[low]) / np.sum(wt * tau_a))
        fb = float(np.sum((wt * tau_b)[low]) / np.sum(wt * tau_b))
        print(f"T = {T:.0f} K: soft sector {sector:.3e} Pa s | stable manifold (a) {eta_a:.3e} "
              f"(x{eta_a / sector:.1f}), (b) {eta_b:.3e} (x{eta_b / sector:.1f}) | share from omega0 < 100 cm-1: "
              f"(a) {fa:.2f}, (b) {fb:.2f} | median tau (a) {np.median(tau_a) * 1e12:.2f} ps, (b) {np.median(tau_b) * 1e12:.2f} ps")
        out_rows.append(f"{T:.0f},{sector:.6e},{eta_a:.6e},{eta_b:.6e},{eta_a / sector:.3f},{eta_b / sector:.3f},{fa:.4f},{fb:.4f}")
    out = DIAG_DIR / "bto_stable_manifold_estimate.csv"
    out.write_text("# bto_stable_manifold_estimate.csv - (bto_stable_manifold_estimate.py)\n"
                   "# ORDER-OF-MAGNITUDE estimate of the BaTiO3 stable-manifold eta_xyxy missing from the zone-\n"
                   "# center-anchored soft sector: bare BaTiO3 frequencies and own shear couplings, lifetimes\n"
                   "# transferred from SrTiO3 (a) or Klemens-scaled (b); see script docstring for the assumptions.\n"
                   "# Production gamma convention throughout (ratio to the sector is convention independent).\n"
                   + "\n".join(out_rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
