#!/usr/bin/env python3
"""Tamura 18O isotope series eta(f) for SrTiO3 at 300 K on the production model (the hybrid model tut_z_od1,
SELF_OFFDIAG = 1, construction C), plus the kinetic-theory and Akhiezer anchors.

Isotope channel: the eigenvector-resolved Tamura rate (Tamura 1983, Eq. 12;
latvisc.isotope.isotope_scattering_rate_projected)

    1/tau_iso(qs) = (pi/2N) omega_qs^2 sum_{q's'} delta(omega_qs - omega_q's')
                    sum_{kappa in O} g2 |e*_kappa(q's') . e_kappa(qs)|^2

with g2 from the exact site-resolved sum (latvisc.isotope.mass_variance_g2, verified against
manuscript Eq. (9)), N = 11^3 q-points, a Gaussian delta of width DOS_SIGMA_CM1, and the oxygen-site
components of the orthonormal polarisation vectors. Linewidths combine by Matthiessen:
Gamma_total = Gamma_anh + Gamma_iso with Gamma_iso = (1/tau_iso)/2.

Eigenvectors and frequencies (production): the SCPH-renormalised modes of the production surface at 300 K
(compute_eta_SrTiO3.construction_modes — omega_r and e_r of the same surface that carries the
couplings). Carried for the ledger: the same series with the bare QE eigenvectors of reference.modes
(frequencies still the renormalised ones), and the former total-DOS form is no longer used.

Anchors (logged as computed): kinetic-theory estimate eta_kin = 3 n_at k_B T <gamma^2 tau> with the
production per-mode gamma_C and tau (plain mode average); Akhiezer conversions at 1 GHz with the
measured shear velocity and density: alpha = omega^2 eta/(2 rho v^3), Q^-1 = omega eta/(rho v^2).

Reads : the production inputs of compute_eta_SrTiO3.py, data/raw/gruneisen_modes/SrTiO3/reference.modes
Writes: data/processed/eta_isotope_SrTiO3.csv
Usage: uv run python scripts/compute_eta_isotope_SrTiO3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR  # noqa: E402
from compute_eta_SrTiO3 import (  # noqa: E402
    CM1, REPO, V_CELL, assemble_construction, construction_modes, export_mode_table, load_construction_surface,
    load_vogt,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import orthonormal_eigenvectors  # noqa: E402
from latvisc.isotope import isotope_scattering_rate_projected, mass_variance_g2  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402
from latvisc.validation import (  # noqa: E402
    akhiezer_attenuation, inverse_quality_factor, kinetic_viscosity_estimate,
)

M_NAT_O = 15.999
M_18O = 17.99916
FRACTIONS = [0.0, 0.01, 0.05, 0.10, 0.15]
T_K = 300
DOS_SIGMA_CM1 = 10.0
O_SITES = [2, 3, 4]                     # atom order Sr, Ti, O, O, O
RHO_MEAS = 5110.0                       # kg/m^3, Bell & Rupprecht 1963
V_SHEAR_MEAS = 4900.0                   # m/s, ~sqrt(c44/rho) transverse, measured scale
F_ACOUSTIC_HZ = 1.0e9
N_Q = 11 ** 3


def bare_o_site_vectors():
    """O-site components of the bare QE eigenvectors, [iq, branch-1, site, xyz] (for the ledger row)."""
    blocks = read_modes(MODES_DIR / "SrTiO3" / "reference.modes")
    z = np.stack([orthonormal_eigenvectors(vec, MASSES["SrTiO3"]) for _, _, vec in blocks])   # (nq, 15, 15)
    return z.reshape(z.shape[0], z.shape[1], 5, 3)[:, :, O_SITES, :]


def projected_rates_per_g2(modes, eigenvectors="renormalised", sigma_cm1=DOS_SIGMA_CM1):
    """Per-mode Tamura rate per unit g2 (rad/s) keyed by (iq, nu), from the renormalised frequencies
    and either the own-surface renormalised eigenvectors or the bare QE eigenvectors of the
    max-overlap partner mode."""
    keys, omegas, evecs = [], [], []
    bare = bare_o_site_vectors() if eigenvectors == "bare" else None
    for m in modes:
        if m["omega_r"] <= 0 or (m["iq"] == 0 and m["omega_r"] < 5.0):
            continue
        keys.append((m["iq"], m["nu"]))
        omegas.append(m["omega_r"])
        evecs.append(m["evec_O"] if bare is None else bare[m["iq"], m["partner"] - 1])
    omegas = np.asarray(omegas) * CM1
    e = np.stack(evecs)
    rate = isotope_scattering_rate_projected(omegas, e, omegas, e, [1.0, 1.0, 1.0], N_Q, sigma_cm1 * CM1)
    return dict(zip(keys, rate))


def isotope_series(eigenvectors="renormalised", fractions=FRACTIONS, sigma_cm1=DOS_SIGMA_CM1, modes=None,
                   surface=None, vogt=None):
    """eta(f)/eta(0) at 300 K on the production model; also returns the f = 0.15 HWHM per mode."""
    if surface is None:
        surface = load_construction_surface(T_K)
    if modes is None:
        modes = construction_modes(T_K, 11, surface)
    if vogt is None:
        vogt, _ = load_vogt()
    per_g2 = projected_rates_per_g2(modes, eigenvectors, sigma_cm1)
    eta0, _, _ = assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt)
    out, extra15 = {}, None
    for f in fractions:
        if f == 0.0:
            out[f] = 1.0
            continue
        g2 = mass_variance_g2([M_NAT_O, M_18O], [1.0 - f, f])
        extra = {k: g2 * v / 2.0 / CM1 for k, v in per_g2.items()}        # HWHM, cm-1
        if abs(f - 0.15) < 1e-9:
            extra15 = extra
        eta, _, _ = assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, extra_gamma_hwhm_cm1=extra)
        out[f] = eta / eta0
    return out, extra15


def main() -> None:
    vogt, _ = load_vogt()
    surface = load_construction_surface(T_K)
    modes = construction_modes(T_K, 11, surface)
    eta0, sec, flags, details = assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, return_details=True)
    series, extra15 = {}, None
    for eig in ("renormalised", "bare"):
        series[eig], ex = isotope_series(eig, modes=modes, surface=surface, vogt=vogt)
        if eig == "renormalised":
            extra15 = ex
    print(f"Tamura 18O series at {T_K} K (exact g2 sum; eigenvector-resolved O-site projection; production model: "
          f"hybrid model, SELF_OFFDIAG = 1, construction C; eta_0 = {eta0:.4e} Pa s)")
    out_rows = ["f_18O,g2,eta_total_Pas,eta_over_eta0,eta_over_eta0_bare_eigenvectors"]
    for f in FRACTIONS:
        g2 = 0.0 if f == 0.0 else mass_variance_g2([M_NAT_O, M_18O], [1.0 - f, f])
        r_prod, r_bare = series["renormalised"][f], series["bare"][f]
        print(f"  f={f:4.2f}: g2={g2:.3e}  eta={eta0 * r_prod:.4e} Pa s  eta/eta(0)={r_prod:.5f}  (bare eigenvectors: {r_bare:.5f})")
        out_rows.append(f"{f},{g2:.6e},{eta0 * r_prod:.6e},{r_prod:.6f},{r_bare:.6f}")
    vals = [series["renormalised"][f] for f in FRACTIONS]
    print(f"eta decreases monotonically with f: {'YES' if all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)) else 'NO'}")

    print(f"\nExternal anchors at {T_K} K (eta_0 = {eta0:.4e} Pa s):")
    gam_sq_tau = [d["gruneisen"] ** 2 * d["tau_s"] for d in details]
    n_at = 5.0 / V_CELL
    eta_kin = kinetic_viscosity_estimate(n_at, T_K, float(np.mean(gam_sq_tau)))
    print(f"  kinetic estimate eta_kin = 3 n_at kB T <gamma^2 tau> = {eta_kin:.4e} Pa s -> eta_full/eta_kin = {eta0 / eta_kin:.2f}")
    omega_ac = 2.0 * np.pi * F_ACOUSTIC_HZ
    alpha = float(akhiezer_attenuation(omega_ac, eta0, RHO_MEAS, V_SHEAR_MEAS))
    q_inv = float(inverse_quality_factor(omega_ac, eta0, RHO_MEAS, V_SHEAR_MEAS))
    print(f"  Akhiezer at 1 GHz (rho={RHO_MEAS:.0f}, v_s={V_SHEAR_MEAS:.0f}): alpha = {alpha:.3e} 1/m = "
          f"{alpha * 8.686e-2:.3f} dB/cm, Q^-1 = {q_inv:.3e}")
    w = np.array([d["eta_contrib"] for d in details])
    g = np.array([d["gruneisen"] for d in details]); tau = np.array([d["tau_s"] for d in details])
    print(f"  zone-rms gamma_C = {np.sqrt(np.mean(g ** 2)):.3f}; largest |gamma_C| = {np.abs(g).max():.2f}; "
          f"eta-weighted <gamma^2> = {np.sum(w * g ** 2) / w.sum():.3f}, eta-weighted <tau> = {np.sum(w * tau) / w.sum() * 1e12:.3f} ps")

    out = REPO / "data" / "processed" / "eta_isotope_SrTiO3.csv"
    header = [
        "# eta_isotope_SrTiO3.csv - produced by scripts/compute_eta_isotope_SrTiO3.py",
        f"# Tamura 18O series at {T_K} K, exact g2 sum (Eq. 9), eigenvector-resolved O-site projection",
        "# (Tamura Eq. 12, rate pi/2N; HWHM = rate/2), Gaussian delta 10 cm-1, Matthiessen",
        "# Gamma_total = Gamma_anh + Gamma_iso. Production model: the hybrid model (tut_z_od1, SELF_OFFDIAG = 1,",
        "# inner mesh 12^3), construction C; eigenvectors and frequencies of that surface on the 11^3 mesh. The bare-QE-",
        "# eigenvector series (same frequencies) is carried for comparison. Mass-variance channel only.",
    ]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(out_rows) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")
    if extra15 is not None:
        export_mode_table(details, modes, REPO / "data" / "processed" / "v3_diagnostics" / "mode_table_SrTiO3_300K_v6.csv",
                          eta0, T_K, extra_gamma=extra15)


if __name__ == "__main__":
    main()
