"""Isotopic mass-disorder scattering.

Tamura mass-variance parameter (per sublattice element):

    g2 = sum_i f_i * (1 - m_i / m_bar)^2,   m_bar = sum_i f_i * m_i

Eigenvector-resolved isotope scattering rate (Tamura 1983, Eq. 12):

    1/tau_iso(qs) = (pi / 2N) omega_qs^2 sum_{q's'} delta(omega_qs - omega_q's')
                    sum_kappa g2(kappa) |e*_kappa(q's') . e_kappa(qs)|^2

with N the number of q-points (unit cells), e_kappa the components of the
orthonormal polarisation vector on site kappa (sum_kappa |e_kappa|^2 = 1) and
g2(kappa) the mass-variance parameter of that site. The half width used
throughout the viscosity pipeline is Gamma_iso = (1/tau_iso)/2.

Polarisation-averaged form (isotropic monatomic limit):

    1/tau_iso(omega) = (pi / 6) * V0 * g2 * omega^2 * dos(omega)

follows from the projected form on a monatomic lattice with the isotropic
average <|e'* . e|^2> = 1/3 and V0 = V_cell. Applied to a polyatomic cell with
V0 = V_cell/n_at and the total density of states it replaces the site overlap
by a constant and is a diagnostic estimate only; it is neither an upper nor a
lower bound of the projected rate (for SrTiO3 it overestimates the rate of the
Sr-dominated low-frequency modes and underestimates it for O-dominated ones).
Channels combine by Matthiessen's rule.

Reference: S. Tamura, Phys. Rev. B 27, 858 (1983).
"""

from __future__ import annotations

import numpy as np

__all__ = ["mass_variance_g2", "isotope_scattering_rate", "isotope_scattering_rate_projected", "matthiessen"]


def mass_variance_g2(masses, fractions):
    """Tamura mass-variance parameter g2 for one crystallographic site.

    Parameters
    ----------
    masses : array_like
        Isotope masses (any consistent unit).
    fractions : array_like
        Isotopic fractions, summing to 1.
    """
    masses = np.asarray(masses, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
        raise ValueError(f"isotope fractions sum to {fractions.sum()}, expected 1")
    mean_mass = np.sum(fractions * masses)
    return float(np.sum(fractions * (1.0 - masses / mean_mass) ** 2))


def isotope_scattering_rate(omega, g2, volume_per_atom, dos):
    """Polarisation-averaged Tamura rate 1/tau_iso in 1/s (monatomic limit).

    Diagnostic estimate only for polyatomic cells; the production rate is
    isotope_scattering_rate_projected.

    Parameters
    ----------
    omega : array_like
        Angular frequencies, rad/s.
    g2 : float
        Mass-variance parameter (dimensionless).
    volume_per_atom : float
        Volume per atom, m^3.
    dos : array_like
        Phonon density of states at omega, states / (m^3 * rad/s).
    """
    omega = np.asarray(omega, dtype=float)
    return (np.pi / 6.0) * volume_per_atom * g2 * omega**2 * np.asarray(dos, dtype=float)


def isotope_scattering_rate_projected(omega, evec_sites, omega_scatterers, evec_sites_scatterers,
                                      g2_sites, n_q, sigma, chunk=400):
    """Eigenvector-resolved Tamura rate 1/tau_iso in 1/s for a set of modes.

    Parameters
    ----------
    omega : (n_modes,) array_like
        Angular frequencies of the modes whose rate is wanted, rad/s.
    evec_sites : (n_modes, n_sites, 3) complex array_like
        Components of the orthonormal polarisation vectors on the sites that
        carry mass variance (sum over ALL sites of |e|^2 = 1 per mode).
    omega_scatterers, evec_sites_scatterers : the same for the modes summed
        over (normally the full mesh), rad/s and (n_scat, n_sites, 3).
    g2_sites : (n_sites,) array_like
        Mass-variance parameter of each site (mass_variance_g2).
    n_q : int
        Number of q-points of the scatterer mesh (unit cells).
    sigma : float
        Gaussian width replacing the delta function, rad/s.

    Returns
    -------
    ndarray (n_modes,) : 1/tau_iso in 1/s (rate; halve for the HWHM).
    """
    omega = np.asarray(omega, dtype=float)
    omega_s = np.asarray(omega_scatterers, dtype=float)
    e = np.asarray(evec_sites, dtype=complex)
    es = np.asarray(evec_sites_scatterers, dtype=complex)
    g2 = np.asarray(g2_sites, dtype=float)
    norm = 1.0 / (float(sigma) * np.sqrt(2.0 * np.pi))
    out = np.zeros(len(omega))
    for start in range(0, len(omega), chunk):
        sl = slice(start, start + chunk)
        weight = np.exp(-0.5 * ((omega[sl][:, None] - omega_s[None, :]) / sigma) ** 2) * norm
        overlap = np.zeros((weight.shape[0], len(omega_s)))
        for k in range(e.shape[1]):
            overlap += g2[k] * np.abs(e[sl, k, :] @ es[:, k, :].conj().T) ** 2
        out[sl] = (np.pi / (2.0 * n_q)) * omega[sl] ** 2 * (weight * overlap).sum(axis=1)
    return out


def matthiessen(*rates):
    """Combine scattering rates (1/s each) by Matthiessen's rule.

    Returns the total rate; the combined lifetime is 1 / total.
    """
    total = np.zeros_like(np.asarray(rates[0], dtype=float))
    for rate in rates:
        total = total + np.asarray(rate, dtype=float)
    return total
