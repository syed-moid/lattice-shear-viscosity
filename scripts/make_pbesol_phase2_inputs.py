#!/usr/bin/env python3
"""Generate PBEsol Phase 2 (harmonic) QE inputs from the Phase 1 relaxes.

Parses dft/qe/<material>/relax_pbesol/vc_relax.out for the converged cell
scale factor, sanity-checks the lattice constant, and writes a single job
directory dft/qe/<material>/dispersion_pbesol/ with the full harmonic
chain (all executed on one VM):

    scf.in        pw.x, relaxed PBEsol cell
    ph_gamma.in   ph.x at Gamma with epsil=.true. (Born charges, eps_inf)
    dynmat.in     dynmat.x, asr='crystal', NAC direction (1,0,0)
    ph_disp.in    ph.x ldisp 4x4x4 grid
    q2r.in        q2r.x, zasr='crystal'
    matdyn.in     matdyn.x along Gamma-X-M-Gamma-R-X (40 pts/segment)

Settings match the audited PBE runs (CALC_SETTINGS.md); only the
functional (via the PBEsol UPFs) and the lattice constant differ.

Usage: uv run python scripts/make_pbesol_phase2_inputs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOHR_TO_ANG = 0.529177210903

MATERIALS = {
    "SrTiO3": {
        "a_atom": ("Sr", 87.62, "Sr.pbesol-spn-kjpaw_psl.1.0.0.UPF"),
        "sanity_ang": (3.88, 3.92),  # literature PBEsol ~3.90 Angstrom
    },
    "BaTiO3": {
        "a_atom": ("Ba", 137.327, "Ba.pbesol-spn-kjpaw_psl.1.0.0.UPF"),
        "sanity_ang": (3.97, 4.01),  # literature PBEsol cubic ~3.99-4.00 Angstrom
    },
}
COMMON = [
    ("Ti", 47.867, "Ti.pbesol-spn-kjpaw_psl.1.0.0.UPF"),
    ("O", 15.999, "O.pbesol-n-kjpaw_psl.1.0.0.UPF"),
]


def relaxed_alat_bohr(material: str) -> float:
    """Final alat = celldm(1) x last CELL_PARAMETERS scale factor."""
    out = (REPO / "dft" / "qe" / material / "relax_pbesol" / "vc_relax.out").read_text()
    celldm = float(re.search(r"celldm\(1\)=\s*([\d.]+)", out).group(1))
    factors = re.findall(r"CELL_PARAMETERS \(alat=\s*([\d.]+)\)\n\s*([\d.]+)", out)
    if not factors:
        raise RuntimeError(f"{material}: no CELL_PARAMETERS blocks in vc_relax.out")
    alat_ref, scale = float(factors[-1][0]), float(factors[-1][1])
    assert abs(alat_ref - celldm) < 1e-6, "celldm mismatch between header and cell block"
    return celldm * scale


def write_inputs(material: str) -> float:
    a_bohr = relaxed_alat_bohr(material)
    a_ang = a_bohr * BOHR_TO_ANG
    low, high = MATERIALS[material]["sanity_ang"]
    status = "OK" if low <= a_ang <= high else "OUTSIDE SANITY WINDOW"
    print(f"{material}: relaxed a = {a_bohr:.6f} bohr = {a_ang:.4f} Angstrom "
          f"[window {low}-{high}: {status}]")
    if status != "OK":
        return a_ang

    a_sym, mass, upf = MATERIALS[material]["a_atom"]
    species = "\n".join(
        f" {s:2s}  {m:<8g} {u}" for s, m, u in [(a_sym, mass, upf)] + COMMON
    )
    positions = f""" {a_sym}  0.000000000  0.000000000  0.000000000
 Ti  0.500000000  0.500000000  0.500000000
 O   0.500000000  0.500000000  0.000000000
 O   0.500000000  0.000000000  0.500000000
 O   0.000000000  0.500000000  0.500000000"""
    prefix = f"{material}_pbesol"
    directory = REPO / "dft" / "qe" / material / "dispersion_pbesol"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "scf.in").write_text(f"""&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = '{prefix}'
    pseudo_dir = './pseudo/'
    outdir = './tmp/'
    tstress = .true.
    tprnfor = .true.
/
&SYSTEM
    ibrav = 1
    celldm(1) = {a_bohr:.6f}           ! PBEsol relaxed (relax_pbesol/vc_relax.out)
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
{species}
ATOMIC_POSITIONS crystal
{positions}
K_POINTS automatic
 8 8 8 0 0 0
""")

    (directory / "ph_gamma.in").write_text(f"""{material} PBEsol -- Gamma phonons with epsil (Born charges + eps_inf)
&inputph
    prefix    = '{prefix}'
    outdir    = './tmp/'
    fildyn    = '{prefix}_gamma.dyn'
    epsil     = .true.
    tr2_ph    = 1.0d-14
    verbosity = 'high'
    recover   = .false.
/
0.0 0.0 0.0
""")

    (directory / "dynmat.in").write_text(f"""&input
    fildyn = '{prefix}_gamma.dyn'
    asr    = 'crystal'
    q(1) = 1.0
    q(2) = 0.0
    q(3) = 0.0
/
""")

    (directory / "ph_disp.in").write_text(f"""{material} PBEsol -- phonon dispersion, 4x4x4 q-grid
&inputph
    prefix    = '{prefix}'
    outdir    = './tmp/'
    fildyn    = '{prefix}.dyn'
    ldisp     = .true.
    nq1 = 4, nq2 = 4, nq3 = 4
    tr2_ph    = 1.0d-14
    verbosity = 'high'
    recover   = .false.
/
""")

    (directory / "q2r.in").write_text(f"""&input
    fildyn = '{prefix}.dyn'
    zasr   = 'crystal'
    flfrc  = '{prefix}.444.fc'
/
""")

    (directory / "matdyn.in").write_text(f"""&input
    flfrc  = '{prefix}.444.fc'
    asr    = 'crystal'
    flfrq  = '{prefix}.freq'
    flvec  = '{prefix}.modes'
    q_in_band_form = .true.
    q_in_cryst_coord = .false.
/
6
  0.0  0.0  0.0   40    ! Gamma
  0.5  0.0  0.0   40    ! X
  0.5  0.5  0.0   40    ! M
  0.0  0.0  0.0   40    ! Gamma
  0.5  0.5  0.5   40    ! R
  0.5  0.0  0.0   1     ! X
""")
    return a_ang


if __name__ == "__main__":
    bad = 0
    for material in sys.argv[1:] or list(MATERIALS):
        a = write_inputs(material)
        low, high = MATERIALS[material]["sanity_ang"]
        bad += not (low <= a <= high)
    sys.exit(1 if bad else 0)
