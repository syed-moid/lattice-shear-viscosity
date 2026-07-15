#!/usr/bin/env python3
"""Generate strained-cell QE inputs for mode Grueneisen tensors (Stage A5).

Strains per material (applied to the relaxed cubic cell, atomic positions
fixed in crystal coordinates — inversion symmetry keeps forces zero under
homogeneous strain, so no internal relaxation occurs):

    shear_xy_p005 / m005 : epsilon_xy = +/-0.5%   (gamma_xy by central diff)
    shear_xy_p010 / m010 : epsilon_xy = +/-1.0%   (step-size convergence)
    hydro_p005  / m005   : isotropic  +/-0.5%     (volumetric gamma check)

Cell vectors are (I + epsilon) . a_i with symmetric strain epsilon. Each
strain directory gets scf.in, ph_disp.in (ldisp 4x4x4, settings identical
to the unstrained dispersion runs), and q2r.in for local post-processing.

Writes: dft/qe/<material>/gruneisen/<strain>/{scf.in,ph_disp.in,q2r.in}

Usage: uv run python scripts/make_strained_inputs.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOHR_TO_ANG = 0.529177210903

MATERIALS = {
    "SrTiO3": {
        "alat_bohr": 7.442023,  # relaxed, dft/qe/SrTiO3/relax/vc_relax.out
        "species": [("Sr", 87.62, "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF")],
    },
    "BaTiO3": {
        "alat_bohr": 7.606726,  # relaxed, dft/qe/BaTiO3/relax/vc_relax.out
        "species": [("Ba", 137.327, "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF")],
    },
}
COMMON_SPECIES = [
    ("Ti", 47.867, "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    ("O", 15.999, "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
]
POSITIONS = """ {A}  0.000000000  0.000000000  0.000000000
 Ti  0.500000000  0.500000000  0.500000000
 O   0.500000000  0.500000000  0.000000000
 O   0.500000000  0.000000000  0.500000000
 O   0.000000000  0.500000000  0.500000000"""

STRAINS = {
    "shear_xy_p005": ("shear", +0.005),
    "shear_xy_m005": ("shear", -0.005),
    "shear_xy_p010": ("shear", +0.010),
    "shear_xy_m010": ("shear", -0.010),
    "hydro_p005": ("hydro", +0.005),
    "hydro_m005": ("hydro", -0.005),
}


def cell_vectors(a_ang: float, kind: str, amplitude: float) -> list[list[float]]:
    if kind == "shear":  # epsilon_xy symmetric shear: (I + eps) columns
        return [
            [a_ang, a_ang * amplitude, 0.0],
            [a_ang * amplitude, a_ang, 0.0],
            [0.0, 0.0, a_ang],
        ]
    scale = 1.0 + amplitude  # hydro
    return [[a_ang * scale, 0.0, 0.0], [0.0, a_ang * scale, 0.0], [0.0, 0.0, a_ang * scale]]


def scf_input(material: str, strain: str, vectors: list[list[float]]) -> str:
    a_species, mass, upf = MATERIALS[material]["species"][0]
    species = [(a_species, mass, upf)] + COMMON_SPECIES
    species_block = "\n".join(f" {s:2s}  {m:<8g} {u}" for s, m, u in species)
    cell_block = "\n".join(f"  {v[0]:.9f}  {v[1]:.9f}  {v[2]:.9f}" for v in vectors)
    return f"""&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = '{material}_{strain}'
    pseudo_dir = './pseudo/'
    outdir = './tmp/'
    tstress = .true.
    tprnfor = .true.
/
&SYSTEM
    ibrav = 0
    nat = 5
    ntyp = 3
    ecutwfc = 60.0
    ecutrho = 480.0
    occupations = 'fixed'
/
&ELECTRONS
    conv_thr = 1.0d-12
    mixing_beta = 0.3
/
ATOMIC_SPECIES
{species_block}
CELL_PARAMETERS angstrom
{cell_block}
ATOMIC_POSITIONS crystal
{POSITIONS.format(A=a_species)}
K_POINTS automatic
 8 8 8 0 0 0
"""


def ph_input(material: str, strain: str) -> str:
    return f"""{material} {strain} -- strained-cell phonon dispersion for mode Grueneisen (4x4x4 grid)
&inputph
    prefix    = '{material}_{strain}'
    outdir    = './tmp/'
    fildyn    = '{material}_{strain}.dyn'
    ldisp     = .true.
    nq1 = 4, nq2 = 4, nq3 = 4
    tr2_ph    = 1.0d-14
    verbosity = 'high'
    recover   = .false.
/
"""


def q2r_input(material: str, strain: str) -> str:
    return f"""&input
    fildyn = '{material}_{strain}.dyn'
    zasr   = 'crystal'
    flfrc  = '{material}_{strain}.444.fc'
/
"""


def main() -> None:
    count = 0
    for material, spec in MATERIALS.items():
        a_ang = spec["alat_bohr"] * BOHR_TO_ANG
        for strain, (kind, amplitude) in STRAINS.items():
            directory = REPO / "dft" / "qe" / material / "gruneisen" / strain
            directory.mkdir(parents=True, exist_ok=True)
            vectors = cell_vectors(a_ang, kind, amplitude)
            (directory / "scf.in").write_text(scf_input(material, strain, vectors))
            (directory / "ph_disp.in").write_text(ph_input(material, strain))
            (directory / "q2r.in").write_text(q2r_input(material, strain))
            count += 1
            print(f"{material}/{strain}: a = {a_ang:.6f} A, {kind} {amplitude:+.3%}")
    print(f"{count} job directories under dft/qe/<material>/gruneisen/")


if __name__ == "__main__":
    main()
