"""Soft-mode treatment near the ferroelectric transition.

Cochran law for the soft TO mode frequency:

    omega_s(T)^2 = cochran_coefficient * (T - T_C)      (T > T_C)

Near T_C the soft mode becomes overdamped (linewidth > omega_s); its
lifetime must then be taken from the overdamped-safe effective form in
`latvisc.viscosity.tau_effective`, which this module re-exports for the
soft-mode workflow.
"""

from __future__ import annotations

import numpy as np

from .viscosity import tau_effective

__all__ = ["cochran_frequency", "is_overdamped", "tau_effective"]


def cochran_frequency(temperature, transition_temperature, cochran_coefficient):
    """Soft-mode angular frequency omega_s(T) from the Cochran law.

    Parameters
    ----------
    temperature : array_like
        Temperature, K. Must be above `transition_temperature`; the soft
        mode of the paraelectric phase is undefined below it.
    transition_temperature : float
        Curie temperature T_C, K.
    cochran_coefficient : float
        Slope of omega_s^2 vs T, (rad/s)^2 / K.

    Returns
    -------
    ndarray
        omega_s in rad/s.
    """
    temperature = np.asarray(temperature, dtype=float)
    reduced = temperature - float(transition_temperature)
    if np.any(reduced < 0.0):
        raise ValueError("cochran_frequency is defined for T >= T_C only")
    return np.sqrt(float(cochran_coefficient) * reduced)


def is_overdamped(omega, linewidth):
    """True where the mode is overdamped (linewidth exceeds omega)."""
    return np.asarray(linewidth, dtype=float) > np.asarray(omega, dtype=float)
