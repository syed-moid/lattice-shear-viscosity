"""Lattice shear viscosity kernel.

Central formula (per-mode sum over phonon branches s and wavevectors q):

    eta_ijlm = (1 / (V k_B T)) * sum_qs (hbar omega)^2
               * gruneisen_ij * gruneisen_lm * n (n + 1) * tau

with the single-mode lifetime tau = 1 / (2 * linewidth). All quantities SI:
omega and linewidth in rad/s (angular), volume in m^3, temperature in K;
eta comes out in Pa s.

High-temperature limit: n(n+1) -> (k_B T / hbar omega)^2, so
eta -> (k_B T / V) * sum(gruneisen^2 * tau).

`gruneisen` (mode Grueneisen tensor component, dimensionless, O(1)) and
`linewidth` (anharmonic broadening, rad/s) are distinct physical quantities
and are never interchangeable.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

__all__ = [
    "thz_to_rad_per_s",
    "bose_einstein",
    "tau_from_linewidth",
    "tau_effective",
    "tau_two_pole_exact",
    "shear_viscosity",
    "shear_viscosity_tensor",
]


def thz_to_rad_per_s(frequency_thz):
    """Convert an ordinary frequency in THz to angular frequency in rad/s."""
    return np.asarray(frequency_thz, dtype=float) * 1e12 * 2.0 * np.pi


def bose_einstein(omega, temperature):
    """Bose-Einstein occupation n(omega, T) for omega in rad/s, T in K.

    Overflow-safe: for hbar*omega >> k_B*T the occupation underflows to 0
    instead of overflowing the exponential.
    """
    omega = np.asarray(omega, dtype=float)
    x = HBAR * omega / (K_B * float(temperature))
    with np.errstate(over="ignore"):
        n = np.where(x > 700.0, np.exp(-np.minimum(x, 745.0)), 1.0 / np.expm1(np.minimum(x, 700.0)))
    return n


def tau_from_linewidth(linewidth):
    """Single-mode lifetime tau = 1 / (2 * linewidth), linewidth in rad/s."""
    linewidth = np.asarray(linewidth, dtype=float)
    return 1.0 / (2.0 * linewidth)


def tau_effective(omega, linewidth):
    """Slow-pole effective lifetime (retained as a limiting form).

    tau_eff = 1 / (2 * [linewidth - Re sqrt(linewidth^2 - omega^2)])

    Underdamped (linewidth < omega): reduces to 1 / (2 * linewidth).
    Overdamped (linewidth > omega): tends to linewidth / omega^2.

    This is the dominant slow-pole contribution to the time-integrated
    two-pole occupation correlator with the full fluctuation weight
    assigned to the slow pole; the exact integral is tau_two_pole_exact,
    which production uses. Both share the deep-overdamped limiting power
    linewidth/omega^2 (up to the factor 2 in the weight).
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    root = np.sqrt((linewidth**2 - omega**2).astype(complex))
    denominator = 2.0 * (linewidth - root.real)
    return 1.0 / denominator


def tau_two_pole_exact(omega, linewidth):
    """Exact time-integrated two-pole (damped-oscillator) lifetime.

    tau = (linewidth^2 + omega^2) / (2 * linewidth * omega^2)
        = 1/(2*linewidth) + linewidth/(2*omega^2)

    Exact closed form of the time-integrated energy-fluctuation
    correlator of the damped oscillator (classical/Wick evaluation of
    the full two-pole spectral form; manuscript Appendix A.3), valid
    uniformly: it reduces to the sharp-resonance 1/(2*linewidth) for
    linewidth << omega and to linewidth/(2*omega^2) deep overdamped,
    with no regime switch. Production formula for every mode.
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    return (linewidth**2 + omega**2) / (2.0 * linewidth * omega**2)


def _mode_weights(omega, temperature):
    """(hbar omega)^2 * n * (n + 1) for each mode."""
    n = bose_einstein(omega, temperature)
    return (HBAR * np.asarray(omega, dtype=float)) ** 2 * n * (n + 1.0)


def shear_viscosity(omega, gruneisen, tau, volume, temperature):
    """Scalar shear viscosity from per-mode arrays.

    Parameters
    ----------
    omega : array_like
        Angular frequencies, rad/s.
    gruneisen : array_like
        Shear-component mode Grueneisen parameters (dimensionless).
    tau : array_like
        Mode lifetimes, s (e.g. from `tau_from_linewidth` or `tau_effective`).
    volume : float
        Crystal volume normalising the mode sum, m^3 (cell volume times the
        number of q-points when summing over a q-mesh).
    temperature : float
        Temperature, K.

    Returns
    -------
    float
        Shear viscosity in Pa s.
    """
    gruneisen = np.asarray(gruneisen, dtype=float)
    weights = _mode_weights(omega, temperature)
    return float(
        np.sum(weights * gruneisen**2 * np.asarray(tau, dtype=float))
        / (volume * K_B * float(temperature))
    )


def shear_viscosity_tensor(omega, gruneisen_ij, gruneisen_lm, tau, volume, temperature):
    """Viscosity tensor component eta_ijlm from two Grueneisen components."""
    weights = _mode_weights(omega, temperature)
    product = np.asarray(gruneisen_ij, dtype=float) * np.asarray(gruneisen_lm, dtype=float)
    return float(
        np.sum(weights * product * np.asarray(tau, dtype=float))
        / (volume * K_B * float(temperature))
    )
