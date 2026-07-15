"""Material parameters for SrTiO3 and BaTiO3.

Lattice constants and cell volumes are from the vc-relax runs committed in
`dft/qe/<material>/relax/vc_relax.out` (PBE, kjpaw psl 1.0.0 pseudos).
Transition temperatures are standard literature values.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Material", "SRTIO3", "BATIO3"]

BOHR_TO_M = 5.29177210903e-11


@dataclass(frozen=True)
class Material:
    name: str
    atoms_per_cell: int
    lattice_constant: float  # m, relaxed cubic cell
    cell_volume: float  # m^3
    mass_density: float  # kg/m^3
    transition_temperature: float  # K


# SrTiO3: relaxed a = 7.370 * 1.009772427 bohr = 3.9383 Angstrom,
# V = 61.077 Angstrom^3 (dft/qe/SrTiO3/relax/vc_relax.out).
# Cubic-to-tetragonal (antiferrodistortive) transition at 105 K; the
# ferroelectric transition is suppressed (quantum paraelectric).
SRTIO3 = Material(
    name="SrTiO3",
    atoms_per_cell=5,
    lattice_constant=7.370 * 1.009772427 * BOHR_TO_M,
    cell_volume=61.07682e-30,
    mass_density=5.11e3,
    transition_temperature=105.0,
)

# BaTiO3: relaxed a = 7.570 * 1.004851461 bohr = 4.0254 Angstrom,
# V = 65.222 Angstrom^3 (dft/qe/BaTiO3/relax/vc_relax.out).
# Cubic-to-tetragonal ferroelectric transition at 393 K.
BATIO3 = Material(
    name="BaTiO3",
    atoms_per_cell=5,
    lattice_constant=7.570 * 1.004851461 * BOHR_TO_M,
    cell_volume=65.22238e-30,
    mass_density=6.02e3,
    transition_temperature=393.0,
)
