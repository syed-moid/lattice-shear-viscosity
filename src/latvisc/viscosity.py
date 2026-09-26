"""Lattice shear viscosity kernel.

Central formula (per-mode sum over phonon branches s and wavevectors q):

    eta_ijlm = (1 / (V k_B T)) * sum_qs (hbar omega)^2
               * gruneisen_ij * gruneisen_lm * n (n + 1) * tau

with the single-mode lifetime tau = 1 / (2 * linewidth) for weakly damped
modes; the production lifetime for every mode is the stress-correlator
kernel tau_two_pole_stress (see below), which reduces to 1 / (2 * linewidth)
when linewidth << omega. All quantities SI: omega and linewidth in rad/s
(angular), volume in m^3, temperature in K; eta comes out in Pa s.

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
    "tau_two_pole_stress",
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
    """Legacy slow-pole effective lifetime (not used on any production path).

    tau_eff = 1 / (2 * [linewidth - Re sqrt(linewidth^2 - omega^2)])

    Underdamped (linewidth < omega): reduces to 1 / (2 * linewidth).
    Overdamped (linewidth > omega): tends to linewidth / omega^2.

    This is the dominant slow-pole contribution to the time-integrated
    two-pole occupation correlator with the full fluctuation weight
    assigned to the slow pole. Kept for comparison only: production uses
    tau_two_pole_stress; tau_two_pole_exact is the energy-variable limit.
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    root = np.sqrt((linewidth**2 - omega**2).astype(complex))
    denominator = 2.0 * (linewidth - root.real)
    return 1.0 / denominator


def tau_two_pole_exact(omega, linewidth):
    """Energy-variable (occupation-diagonal) two-pole lifetime.

    tau = (linewidth^2 + omega^2) / (2 * linewidth * omega^2)
        = 1/(2*linewidth) + linewidth/(2*omega^2)

    Exact classical time integral of the ENERGY-fluctuation correlator of
    a damped harmonic oscillator with friction 2*linewidth (Wick
    evaluation of the two-pole form), i.e. of the a^dag a-diagonal
    (DeVault) stress with the a a and a^dag a^dag terms dropped. It
    reduces to 1/(2*linewidth) for linewidth << omega and to
    linewidth/(2*omega^2) deep overdamped. Retained for comparison with
    the production kernel tau_two_pole_stress, from which it differs by a
    factor that grows from 1 (underdamped) to 4 (deep overdamped).
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    return (linewidth**2 + omega**2) / (2.0 * linewidth * omega**2)


def tau_two_pole_stress(omega, linewidth):
    """Stress-correlator two-pole lifetime (production kernel).

    tau = 1/(2*linewidth) + 2*linewidth/omega^2

    Exact classical time integral of the correlator of the
    stiffness-conjugate stress S = dH/d eps = (1/2)(d omega^2/d eps) x^2 of
    an effective damped harmonic oscillator x'' + 2*linewidth*x' + omega^2 x
    = noise, normalised to the classical weight gamma^2 (k_B T)^2 that the
    mode sum carries through (hbar omega)^2 gamma^2 n(n+1). The x^2
    operator keeps the a a and a^dag a^dag content that the energy variable
    drops; those terms oscillate at 2 omega and integrate to zero when
    linewidth << omega, where this kernel reduces to 1/(2*linewidth), and
    they dominate when the oscillator no longer oscillates (deep overdamped
    limit 2*linewidth/omega^2, four times the energy-variable value).
    Quantum corrections to the classical integral, with the Bose factors
    kept inside the spectral convolution, are below 0.2 % for every SrTiO3
    mode at 300 K and below 7 % up to linewidth/omega = 3, hbar omega/k_B T
    = 0.9. Units as tau_two_pole_exact (rad/s in, s out); the linewidth is
    the oscillator friction parameter, equal to the HWHM for underdamped
    modes.
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    return 1.0 / (2.0 * linewidth) + 2.0 * linewidth / omega**2


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
