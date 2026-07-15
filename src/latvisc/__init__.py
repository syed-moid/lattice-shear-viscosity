"""latvisc: lattice shear viscosity of strongly anharmonic ferroelectric oxides.

Central formula (SI units throughout; eta in Pa s):

    eta_ijlm = (1 / (V k_B T)) * sum_qs (hbar omega)^2
               * gruneisen_ij * gruneisen_lm * n (n + 1) * tau
"""

from .gruneisen import mode_gruneisen_finite_strain, mode_gruneisen_volume
from .isotope import isotope_scattering_rate, mass_variance_g2, matthiessen
from .materials import BATIO3, SRTIO3, Material
from .softmode import cochran_frequency, is_overdamped
from .validation import (
    akhiezer_attenuation,
    inverse_quality_factor,
    kinetic_viscosity_estimate,
)
from .viscosity import (
    bose_einstein,
    shear_viscosity,
    shear_viscosity_tensor,
    tau_effective,
    tau_from_linewidth,
    thz_to_rad_per_s,
)

__all__ = [
    "BATIO3",
    "SRTIO3",
    "Material",
    "akhiezer_attenuation",
    "bose_einstein",
    "cochran_frequency",
    "inverse_quality_factor",
    "is_overdamped",
    "isotope_scattering_rate",
    "kinetic_viscosity_estimate",
    "mass_variance_g2",
    "matthiessen",
    "mode_gruneisen_finite_strain",
    "mode_gruneisen_volume",
    "shear_viscosity",
    "shear_viscosity_tensor",
    "tau_effective",
    "tau_from_linewidth",
    "thz_to_rad_per_s",
]
