"""Isotopic mass-disorder scattering.

Tamura mass-variance parameter (per sublattice element):

    g2 = sum_i f_i * (1 - m_i / m_bar)^2,   m_bar = sum_i f_i * m_i

Point-defect (isotope) scattering rate in the Tamura model:

    1/tau_iso(omega) = (pi / 6) * V0 * g2 * omega^2 * dos(omega)

with V0 the volume per atom and dos the phonon density of states per unit
volume and unit angular frequency. Channels combine by Matthiessen's rule.

Reference: S. Tamura, Phys. Rev. B 27, 858 (1983).
"""

from __future__ import annotations

import numpy as np

__all__ = ["mass_variance_g2", "isotope_scattering_rate", "matthiessen"]


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
    """Tamura isotope scattering rate 1/tau_iso in 1/s.

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


def matthiessen(*rates):
    """Combine scattering rates (1/s each) by Matthiessen's rule.

    Returns the total rate; the combined lifetime is 1 / total.
    """
    total = np.zeros_like(np.asarray(rates[0], dtype=float))
    for rate in rates:
        total = total + np.asarray(rate, dtype=float)
    return total
