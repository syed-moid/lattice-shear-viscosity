"""Single source of truth for matdyn.x q-point conventions.

Shear strain distorts the reciprocal lattice, so a fixed Cartesian q-point
does not track the same physical crystal momentum across differently
strained cells: comparing omega(q_cartesian) between +eps/-eps/0 picks up
spurious d(omega)/dq * delta_q(strain) band-slope contamination on top of
any real anharmonic shift. Crystal (fractional) q-coordinates are the
correct convention for cross-strain mode comparison, because a fixed
fractional label tracks the same crystal momentum under smooth lattice
deformation. Every matdyn.x input used anywhere in the Grueneisen pipeline
(the free nonlinearity diagnostic and the production mesh alike) MUST be
built through write_matdyn_input below, so this convention can never
diverge between the two call sites again.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["gamma_centered_mesh", "write_matdyn_input"]


def gamma_centered_mesh(n: int = 11) -> list[tuple[float, float, float]]:
    """n x n x n Gamma-centered fractional q-mesh, i/n for i in 0..n-1.

    Matches the q2r/matdyn convention already used for the production
    dispersion mesh (see dft/qe/mesh_audit/matdyn_mesh_*.in).
    """
    step = 1.0 / n
    return [
        (i * step, j * step, k * step)
        for i in range(n)
        for j in range(n)
        for k in range(n)
    ]


def write_matdyn_input(path, flfrc, flfrq, flvec, qpoints, asr: str = "crystal") -> None:
    """Write a matdyn.x &input namelist in fractional (crystal) q-coordinates.

    Cartesian/band-form input is intentionally not supported here: it
    silently breaks strained-cell mode continuity (see the G1' incident
    log in data/processed/reports/GATE_1p.md).
    """
    qpoints = list(qpoints)
    lines = [
        "&input",
        f"    flfrc  = '{flfrc}'",
        f"    asr    = '{asr}'",
        f"    flfrq  = '{flfrq}'",
        f"    flvec  = '{flvec}'",
        "    q_in_band_form = .false.",
        "    q_in_cryst_coord = .true.",
        "/",
        str(len(qpoints)),
    ]
    lines.extend(f"  {qx:.9f}  {qy:.9f}  {qz:.9f}" for qx, qy, qz in qpoints)
    Path(path).write_text("\n".join(lines) + "\n")
