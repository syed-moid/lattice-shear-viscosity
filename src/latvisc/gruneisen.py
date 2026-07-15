"""Finite-strain mode Grueneisen parameters.

The mode Grueneisen tensor component for mode (q, s) and strain epsilon_ij is

    gruneisen_ij(q, s) = - (1 / omega(q, s)) * d omega(q, s) / d epsilon_ij

evaluated here by central finite differences from two strained phonon runs
(+delta and -delta applied to the same strain component).
"""

from __future__ import annotations

import numpy as np

__all__ = ["mode_gruneisen_finite_strain", "mode_gruneisen_volume"]


def mode_gruneisen_finite_strain(omega_reference, omega_plus, omega_minus, strain_amplitude):
    """Central-difference mode Grueneisen parameter per mode.

    Parameters
    ----------
    omega_reference : array_like
        Unstrained mode frequencies (any consistent unit).
    omega_plus, omega_minus : array_like
        Mode frequencies at strain +delta and -delta. Branches must be
        matched (same mode ordering) across the three runs.
    strain_amplitude : float
        The strain delta (dimensionless).

    Returns
    -------
    ndarray
        Dimensionless gruneisen_ij per mode, expected O(1).
    """
    omega_reference = np.asarray(omega_reference, dtype=float)
    domega = (np.asarray(omega_plus, dtype=float) - np.asarray(omega_minus, dtype=float)) / (
        2.0 * float(strain_amplitude)
    )
    return -domega / omega_reference


def mode_gruneisen_volume(omega_plus, omega_minus, volume_plus, volume_minus):
    """Volumetric (thermodynamic) mode Grueneisen from two volume-scaled runs.

    gruneisen = - d ln omega / d ln V, central difference.
    """
    dlnomega = np.log(np.asarray(omega_plus, dtype=float)) - np.log(np.asarray(omega_minus, dtype=float))
    dlnvolume = np.log(float(volume_plus)) - np.log(float(volume_minus))
    return -dlnomega / dlnvolume
