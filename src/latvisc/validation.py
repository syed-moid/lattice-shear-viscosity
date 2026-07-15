"""Independent cross-checks on the computed viscosity.

Kinetic-theory order-of-magnitude estimate:

    eta ~ 3 * n_at * k_B * T * <gruneisen^2 * tau>

(3 modes per atom; n_at the atomic number density). In the classical limit
this coincides with the full mode sum, so agreement within a factor of a
few at 300 K is a required consistency check.

Akhiezer conversions relating viscosity to ultrasonic attenuation:

    alpha = omega^2 * eta / (2 * rho * v^3)      (amplitude attenuation, 1/m)
    Q^-1  = 2 * alpha * v / omega                (internal friction)
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as K_B

__all__ = [
    "kinetic_viscosity_estimate",
    "akhiezer_attenuation",
    "inverse_quality_factor",
]


def kinetic_viscosity_estimate(atom_density, temperature, mean_gruneisen_sq_tau):
    """Kinetic-theory estimate eta ~ 3 n_at k_B T <gruneisen^2 tau> in Pa s.

    Parameters
    ----------
    atom_density : float
        Atomic number density n_at, 1/m^3.
    temperature : float
        Temperature, K.
    mean_gruneisen_sq_tau : float
        Mode average of gruneisen^2 * tau, s.
    """
    return 3.0 * atom_density * K_B * temperature * mean_gruneisen_sq_tau


def akhiezer_attenuation(omega, viscosity, mass_density, sound_velocity):
    """Akhiezer amplitude attenuation alpha = omega^2 eta / (2 rho v^3), 1/m."""
    omega = np.asarray(omega, dtype=float)
    return omega**2 * viscosity / (2.0 * mass_density * sound_velocity**3)


def inverse_quality_factor(omega, viscosity, mass_density, sound_velocity):
    """Internal friction Q^-1 = 2 alpha v / omega = omega eta / (rho v^2)."""
    alpha = akhiezer_attenuation(omega, viscosity, mass_density, sound_velocity)
    return 2.0 * alpha * sound_velocity / np.asarray(omega, dtype=float)
