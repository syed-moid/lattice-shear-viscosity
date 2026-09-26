#!/usr/bin/env python3
"""Audit 0.B: Tamura isotope rate with the exact oxygen-site eigenvector
projection, compared with the production total-DOS form.

Three forms of the mass-variance (Tamura 1983, Eq. 12) rate, all HWHM
Gamma_iso = (1/tau_iso)/2 downstream:

  (i)   manuscript Eq. (10) as printed:
        Gamma_iso = (pi/12N) w^2 sum_{q's'} delta(w - w') sum_k g2(k) |e*_k(q's').e_k(qs)|^2
  (ii)  Tamura eigenvector-resolved rate (this script):
        1/tau_iso = (pi/2N) w^2 sum_{q's'} delta(w - w') sum_k g2(k) |e*_k(q's').e_k(qs)|^2
        so Gamma_iso = (pi/4N) w^2 sum ...  -- the printed (i) is a factor 3 low.
  (iii) production code (latvisc.isotope.isotope_scattering_rate):
        1/tau_iso = (pi/6) V0 g2 w^2 dos_total(w),  V0 = V_cell/5,
        dos_total = (1/(V_cell N)) sum_{q's'} delta(w - w'),
        i.e. 1/tau_iso = (pi/30N) g2 w^2 sum_{q's'} delta(w - w').
        (ii) -> (iii) on a MONATOMIC lattice by the isotropic polarisation
        average <|e'*.e|^2> = 1/3 (V0 = V_cell, one site). For the 5-atom
        cell with the variance on the three O sites, (iii) amounts to
        replacing sum_{k in O} |e*_k'.e_k|^2 by the constant 1/15 for every
        pair of modes; the exact projection replaces that constant by the
        actual O-site overlap of the bare (QE) eigenvectors.

Eigenvectors: data/raw/gruneisen_modes/SrTiO3/reference.modes (matdyn flvec
on the production 11^3 fractional mesh, bare PBEsol harmonic), mass-weighted
and normalised by latvisc.gruneisen.orthonormal_eigenvectors, so that
sum_k |e_k|^2 = 1 per mode (Tamura's normalisation). Frequencies in the
delta function and the w^2 prefactor: the same mapped renormalised
frequencies the production DOS uses (build_dos), so that the ONLY change
between (iii) and (ii) is the site projection. Gaussian smearing sigma =
10 cm-1 (production), plus 5 and 20 cm-1. g2 from the exact site sum
(latvisc.isotope.mass_variance_g2), same Matthiessen composition and the
production two-pole kernel.

Sign check (0.B.4): with tau_two_pole_exact, d tau/d Gamma = -1/(2 Gamma^2)
+ 1/(2 w^2) is positive for Gamma > w; the eta weight sitting in such
modes and the sign of the total trend are reported from the full sum.

Writes: data/processed/v3_diagnostics/isotope_rates_SrTiO3_300K.csv (per mode)
        data/processed/v3_diagnostics/isotope_series_exact_projection.csv
"""

from __future__ import annotations

import numpy as np

from diag_common import (CM1, MASSES, MODES_DIR, N_Q, REPO, DIAG_DIR, eta_mod, load_rows,
                          mode_table, recompute_eta, tau_two_pole_exact)
from compute_eta_isotope_SrTiO3 import V0, build_dos  # noqa: E402
from latvisc.gruneisen import orthonormal_eigenvectors  # noqa: E402
from latvisc.isotope import isotope_scattering_rate, mass_variance_g2  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402

M_NAT_O, M_18O = 15.999, 17.99916
FRACTIONS = [0.01, 0.05, 0.10, 0.15]
SIGMAS_CM1 = [5.0, 10.0, 20.0]
O_SITES = [2, 3, 4]          # atom order Sr, Ti, O, O, O (make_strained_inputs.POSITIONS)
CHUNK = 400


def load_eigenvectors():
    """Orthonormal polarisation vectors, shape (n_q*15, 15); index m = iq*15 + (branch-1)."""
    blocks = read_modes(MODES_DIR / "SrTiO3" / "reference.modes")
    return np.concatenate([orthonormal_eigenvectors(vec, MASSES["SrTiO3"]) for _, _, vec in blocks])


def main() -> None:
    eta0, sec, flags, details = mode_table(300)
    print(f"production eta(300 K) from details: {eta0:.6e} Pa s over {len(details)} modes")
    z = load_eigenvectors()

    # scatterer set: same as production build_dos (all non-acoustic modes with mapped lambda_r > 0)
    map_lambda = eta_mod.build_maps(300)[0]
    rows = load_rows("SrTiO3")
    scat_idx, scat_w = [], []
    for r in rows:
        if r["acoustic"]:
            continue
        lam = map_lambda(np.sign(r["omega_ref"]) * r["omega_ref"] ** 2)
        if lam > 0:
            scat_idx.append(r["iq"] * 15 + r["branch"] - 1)
            scat_w.append(np.sqrt(lam))
    scat_idx = np.asarray(scat_idx)
    scat_w = np.asarray(scat_w)                    # cm-1, mapped renormalised
    print(f"scatterers: {len(scat_idx)} modes (production DOS set); eigenvector array {z.shape}")

    m_idx = np.array([d["iq"] * 15 + d["branch"] - 1 for d in details])
    w_r = np.array([d["omega_r"] for d in details])          # cm-1 (Vogt/floor applied, as production)

    e_o = [z[:, 3 * k:3 * k + 3] for k in O_SITES]
    e_o_scat = [e[scat_idx] for e in e_o]
    o_weight = sum(np.sum(np.abs(e) ** 2, axis=1) for e in e_o)   # O-sublattice share of each mode

    geo_exact = {s: np.zeros(len(details)) for s in SIGMAS_CM1}   # sum_m' W P  (per unit g2)
    geo_dos = {s: np.zeros(len(details)) for s in SIGMAS_CM1}     # sum_m' W / 15
    for start in range(0, len(details), CHUNK):
        sl = slice(start, start + CHUNK)
        zc = m_idx[sl]
        P = np.zeros((len(zc), len(scat_idx)))
        for e, es in zip(e_o, e_o_scat):
            P += np.abs(e[zc] @ es.conj().T) ** 2
        for s in SIGMAS_CM1:
            W = np.exp(-0.5 * ((w_r[sl][:, None] - scat_w[None, :]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            geo_exact[s][sl] = (W * P).sum(axis=1)
            geo_dos[s][sl] = W.sum(axis=1) / 15.0

    def rate_per_g2(geo):
        """rate (rad/s) per unit g2: (pi/2N) w^2 * sum_m' delta(w-w') P, delta in 1/(rad/s)."""
        return (np.pi / (2.0 * N_Q)) * (w_r * CM1) ** 2 * geo / CM1

    # cross-check of the DOS form against the production implementation (sigma = 10)
    dos = build_dos(rows)
    g2_ref = mass_variance_g2([M_NAT_O, M_18O], [0.85, 0.15])
    prod_rate = np.array([float(np.squeeze(isotope_scattering_rate(w * CM1, g2_ref, V0, dos(w * CM1))))
                          for w in w_r[:200]])
    mine = rate_per_g2(geo_dos[10.0])[:200] * g2_ref
    print(f"DOS-form rate, this script vs production (first 200 modes): max rel dev "
          f"{np.max(np.abs(mine / prod_rate - 1)):.2e}")

    ratio = geo_exact[10.0] / geo_dos[10.0]
    weights = np.array([d["eta_contrib"] for d in details]) / eta0
    print("\nper-mode ratio Gamma_iso^exact / Gamma_iso^DOS (sigma = 10 cm-1):")
    print(f"  unweighted: median {np.median(ratio):.3f}, mean {ratio.mean():.3f}, "
          f"p10 {np.percentile(ratio, 10):.3f}, p90 {np.percentile(ratio, 90):.3f}, "
          f"min {ratio.min():.3f}, max {ratio.max():.3f}")
    print(f"  eta-weighted mean: {np.sum(weights * ratio):.3f}")
    print(f"  O-sublattice share of eigenvector norm, eta-weighted: {np.sum(weights * o_weight[m_idx]):.3f}")
    for lo, hi in [(0, 50), (50, 100), (100, 175), (175, 300), (300, 600), (600, 1000)]:
        m = (w_r >= lo) & (w_r < hi)
        if m.any():
            print(f"    omega_r in [{lo},{hi}) cm-1: n={m.sum():5d}  median ratio {np.median(ratio[m]):.3f}  "
                  f"eta share {weights[m].sum():.3f}  weighted ratio "
                  f"{np.sum(weights[m] * ratio[m]) / max(weights[m].sum(), 1e-30):.3f}")

    lines = ["iq,branch,sector,omega0_cm1,omega_r_cm1,Gamma_anh_hwhm_cm1,eta_weight,O_share,"
             "rate_dos_per_g2_rad_s,rate_exact_per_g2_rad_s,ratio_exact_over_dos_s10,"
             "ratio_exact_over_dos_s5,ratio_exact_over_dos_s20"]
    rd10, re10 = rate_per_g2(geo_dos[10.0]), rate_per_g2(geo_exact[10.0])
    r5 = geo_exact[5.0] / geo_dos[5.0]
    r20 = geo_exact[20.0] / geo_dos[20.0]
    for i, d in enumerate(details):
        lines.append(f"{d['iq']},{d['branch']},{d['sector']},{d['omega0']:.4f},{d['omega_r']:.4f},"
                     f"{d['gamma_hwhm']:.5f},{weights[i]:.4e},{o_weight[m_idx[i]]:.4f},"
                     f"{rd10[i]:.5e},{re10[i]:.5e},{ratio[i]:.4f},{r5[i]:.4f},{r20[i]:.4f}")
    out = DIAG_DIR / "isotope_rates_SrTiO3_300K.csv"
    out.write_text("# isotope_rates_SrTiO3_300K.csv - .B (isotope_exact_projection.py)\n"
                   "# Per-mode Tamura rate per unit g2, DOS form (production, pi/6 V0 g2 w^2 dos_total)\n"
                   "# vs exact O-site eigenvector projection (pi/2N w^2 sum delta sum_O |e*.e|^2), 300 K,\n"
                   "# mapped renormalised frequencies, Gaussian smearing sigma = 10 cm-1 (5/20 for the ratio columns).\n"
                   "# HWHM Gamma_iso = g2 * rate/2. Mode index: q index (0-based, 11^3 fractional mesh), branch 1-based.\n"
                   + "\n".join(lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")

    print("\neta(f)/eta(0) at 300 K, production kernel:")
    srows = ["form,sigma_cm1,f_18O,g2,eta_Pas,eta_over_eta0,delta_pct_from_modes_Gamma_lt_omega,"
             "delta_pct_from_modes_Gamma_gt_omega"]
    gam_anh = np.array([d["gamma_hwhm"] for d in details])
    overd = gam_anh > w_r
    print(f"  modes with Gamma_anh > omega_r at 300 K: {overd.sum()} of {len(details)}, eta weight "
          f"{weights[overd].sum():.4%}")
    base_c = np.array([d["eta_contrib"] for d in details])
    for form, geo in [("dos", geo_dos), ("exact", geo_exact)]:
        for s in SIGMAS_CM1:
            for f in FRACTIONS:
                g2 = mass_variance_g2([M_NAT_O, M_18O], [1 - f, f])
                giso_cm1 = g2 * rate_per_g2(geo[s]) / 2.0 / CM1          # HWHM, cm-1
                eta_f, _, c = recompute_eta(details, 300, tau_two_pole_exact, gamma_iso_hwhm_cm1=giso_cm1)
                d_lt = (c - base_c)[~overd].sum() / eta0 * 100
                d_gt = (c - base_c)[overd].sum() / eta0 * 100
                srows.append(f"{form},{s},{f},{g2:.6e},{eta_f:.6e},{eta_f / eta0:.6f},{d_lt:+.4f},{d_gt:+.4f}")
                if s == 10.0 or f == 0.15:
                    print(f"  {form:5s} sigma={s:4.0f} f={f:.2f}: eta = {eta_f:.5e}  eta/eta0 = {eta_f / eta0:.5f}  "
                          f"(dEta from Gamma<w modes {d_lt:+.3f}%, from Gamma>w modes {d_gt:+.4f}%)")
    out = DIAG_DIR / "isotope_series_exact_projection.csv"
    out.write_text("# isotope_series_exact_projection.csv - .B (isotope_exact_projection.py)\n"
                   "# 18O series at 300 K recomputed from the production per-mode details with the DOS-form\n"
                   "# rate (must reproduce data/processed/eta_isotope_SrTiO3.csv at sigma=10) and with the exact\n"
                   "# O-site projected Tamura rate. Last two columns split the change in eta between modes with\n"
                   "# Gamma_anh < omega_r (d tau/d Gamma < 0) and Gamma_anh > omega_r (d tau/d Gamma > 0).\n"
                   + "\n".join(srows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
