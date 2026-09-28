"""Projected (eigenvector-resolved) Tamura rate.

1. On a synthetic isotropic monatomic lattice (one site, unit 3-vectors
   uniformly distributed, all scatterers at one frequency) the projected
   rate equals the polarisation-averaged DOS form to the statistical
   precision of <|e'*.e|^2> = 1/3.
2. If every scatterer shares the mode's polarisation the projected rate is
   exactly 3x the DOS form (|e.e|^2 = 1 instead of 1/3).
3. The SrTiO3 series on the production surface (tutorial harmonic set +
   distributed SCPH corrections, direct eigenvector-matched omega_r) pins the
   regenerated values, bare and renormalised eigenvectors within 2e-4 of each
   other, when the raw mode files are present (skipped otherwise).
"""

from pathlib import Path

import numpy as np
import pytest

from latvisc.isotope import isotope_scattering_rate, isotope_scattering_rate_projected

REPO = Path(__file__).resolve().parent.parent


def _dos_form(omega, g2, n_q, sigma, omega_s, v_cell):
    dos = np.exp(-0.5 * ((omega - omega_s) / sigma) ** 2).sum() / (sigma * np.sqrt(2 * np.pi) * v_cell * n_q)
    return isotope_scattering_rate(omega, g2, v_cell, dos)


def test_isotropic_monatomic_limit():
    rng = np.random.default_rng(7)
    n = 20000
    e = rng.normal(size=(n, 3))
    e /= np.linalg.norm(e, axis=1)[:, None]
    omega_s = np.full(n, 1.0e13)
    omega = np.array([1.0e13])
    g2, sigma, v_cell = 1e-3, 1e12, 6e-29
    proj = isotope_scattering_rate_projected(omega, e[:1][:, None, :], omega_s, e[:, None, :], [g2], n, sigma)
    dos = _dos_form(omega, g2, n, sigma, omega_s, v_cell)
    assert proj[0] == pytest.approx(float(np.squeeze(dos)), rel=0.03)


def test_aligned_polarisations_give_three_times_dos_form():
    n = 500
    e = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))
    omega_s = np.linspace(0.9e13, 1.1e13, n)
    omega = np.array([1.0e13])
    g2, sigma, v_cell = 2e-3, 3e11, 6e-29
    proj = isotope_scattering_rate_projected(omega, e[:1][:, None, :], omega_s, e[:, None, :], [g2], n, sigma)
    dos = _dos_form(omega, g2, n, sigma, omega_s, v_cell)
    assert proj[0] == pytest.approx(3.0 * float(np.squeeze(dos)), rel=1e-9)


@pytest.mark.skipif(not (REPO / "data" / "raw" / "gruneisen_modes" / "SrTiO3" / "reference.modes").exists(),
                    reason="raw mode files not present")
def test_production_series_pinned():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import compute_eta_isotope_SrTiO3 as iso
    # regenerated on the production model (hybrid model tut_z_od1: SELF_OFFDIAG = 1, correction mesh 2^3, inner mesh 12^3,
    # construction C, production-surface eigenvectors and frequencies at 300 K); the previous pins on the own-surface model
    # (inner mesh 2^3) were 0.99913/0.99605/0.99291/0.99038 (renormalised) and 0.99919/0.99632/0.99341/0.99108 (bare)
    # re-pinned after the exact-degeneracy coupling rule and the degenerate linewidth averaging (v3.1); the v3.0 pins were
    # 0.999572/0.998059/0.996519/0.995281 (renormalised) and 0.999567/0.998034/0.996469/0.995208 (bare); re-pinned again after
    # the translationally invariant (ALAMODE) convention of the harmonic strained sets (bare pins were
    # 0.999555/0.997978/0.996370/0.995074, renormalised 0.995161 at f = 0.15)
    ref_renorm = {0.01: 0.999561, 0.05: 0.998008, 0.10: 0.996429, 0.15: 0.995160}
    ref_bare = {0.01: 0.999557, 0.05: 0.997987, 0.10: 0.996387, 0.15: 0.995098}
    surface = iso.load_construction_surface(iso.T_K)
    modes = iso.construction_modes(iso.T_K, 11, surface)
    s_ren, _ = iso.isotope_series(eigenvectors="renormalised", fractions=list(ref_renorm), modes=modes, surface=surface)
    s_bare, _ = iso.isotope_series(eigenvectors="bare", fractions=list(ref_bare), modes=modes, surface=surface)
    for f in ref_renorm:
        assert s_ren[f] == pytest.approx(ref_renorm[f], abs=2e-5)
        assert s_bare[f] == pytest.approx(ref_bare[f], abs=2e-5)
        assert abs(s_ren[f] - s_bare[f]) < 1e-3
