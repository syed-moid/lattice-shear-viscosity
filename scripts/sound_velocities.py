#!/usr/bin/env python3
"""Extract sound velocities from the harmonic dispersion CSVs.

Fits the three acoustic branches near Gamma along [100] (Gamma-X segment),
[110] (M-Gamma segment, approached from the Gamma side), and [111]
(Gamma-R segment) and converts slopes to m/s using the relaxed lattice
constants. Writes data/processed/sound_velocities.csv including literature
comparison values with citations.

The acoustic branches at each small-q point are identified as the three
branches with the smallest |omega| (the unstable soft branches have large
|omega| and sort well away from them).

Usage: uv run python scripts/sound_velocities.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
C_LIGHT_CM_S = 2.99792458e10
BOHR_TO_M = 5.29177210903e-11

RELAX_DIRS = {"pbe": "relax", "pbesol": "relax_pbesol"}


def relaxed_lattice_m(material: str, functional: str) -> float:
    """Final alat from the vc-relax output of the given functional."""
    import re
    out = (REPO / "dft" / "qe" / material / RELAX_DIRS[functional] / "vc_relax.out").read_text()
    celldm = float(re.search(r"celldm\(1\)=\s*([\d.]+)", out).group(1))
    factors = re.findall(r"CELL_PARAMETERS \(alat=\s*([\d.]+)\)\n\s*([\d.]+)", out)
    return celldm * float(factors[-1][1]) * BOHR_TO_M
CELL_MASS_AMU = {"SrTiO3": 87.62 + 47.867 + 3 * 15.999, "BaTiO3": 137.327 + 47.867 + 3 * 15.999}
AMU = 1.66053906660e-27

# (segment used, path indices approaching Gamma in order of increasing |q|,
#  Gamma reference index)
DIRECTIONS = {
    "[100]": (0, [1, 2, 3, 4]),
    "[110]": (120, [119, 118, 117, 116]),
    "[111]": (120, [121, 122, 123, 124]),
}

# Literature single-crystal elastic constants (Pa) and densities (kg/m^3).
# SrTiO3 300 K: Bell and Rupprecht, Phys. Rev. 129, 90 (1963).
# BaTiO3 cubic phase (~443 K, Brillouin): Li, Chan, Grimsditch, Zouboulis,
# J. Appl. Phys. 70, 7327 (1991). Values as commonly quoted; page-check
# both originals before manuscript use.
LITERATURE = {
    "SrTiO3": {
        "citation": "Bell and Rupprecht, Phys. Rev. 129, 90 (1963), 300 K",
        "C11": 317.6e9,
        "C12": 102.5e9,
        "C44": 123.5e9,
        "density": 5110.0,
    },
    "BaTiO3": {
        "citation": "Li, Chan, Grimsditch, Zouboulis, J. Appl. Phys. 70, 7327 (1991), cubic ~443 K",
        "C11": 173.0e9,
        "C12": 82.0e9,
        "C44": 108.0e9,
        "density": 5990.0,
    },
}


def literature_velocities(mat: str) -> dict:
    lit = LITERATURE[mat]
    rho = lit["density"]
    return {
        ("[100]", "T"): np.sqrt(lit["C44"] / rho),
        ("[100]", "L"): np.sqrt(lit["C11"] / rho),
        ("[110]", "T"): np.sqrt((lit["C11"] - lit["C12"]) / 2.0 / rho),  # pol [1-10]
        ("[110]", "T2"): np.sqrt(lit["C44"] / rho),  # pol [001]
        ("[110]", "L"): np.sqrt((lit["C11"] + lit["C12"] + 2 * lit["C44"]) / 2.0 / rho),
        ("[111]", "T"): np.sqrt((lit["C11"] - lit["C12"] + lit["C44"]) / 3.0 / rho),
        ("[111]", "L"): np.sqrt((lit["C11"] + 2 * lit["C12"] + 4 * lit["C44"]) / 3.0 / rho),
    }


def acoustic_velocities(mat: str, functional: str) -> list[dict]:
    df = pd.read_csv(REPO / "data" / "processed" / f"harmonic_dispersion_{mat}.csv", comment="#")
    df = df[df.functional == functional]
    if df.empty:
        return []
    a = relaxed_lattice_m(mat, functional)
    rows = []
    lit_v = literature_velocities(mat)
    for direction, (gamma_index, indices) in DIRECTIONS.items():
        q_gamma = df[df.path_index == gamma_index][["qx", "qy", "qz"]].iloc[0].values
        # slopes fitted through the origin: omega = v * |q|
        q_norms, branch_freqs = [], []
        for i in indices:
            block = df[df.path_index == i].sort_values("branch")
            q = block[["qx", "qy", "qz"]].iloc[0].values
            q_si = np.linalg.norm(q - q_gamma) * 2.0 * np.pi / a
            freqs = block.omega_cm1.values
            # acoustic branches are the three smallest POSITIVE frequencies:
            # near Gamma they are strictly positive, while soft optic branches
            # are imaginary (negative) and must not enter the fit
            positive = freqs[freqs > 0.0]
            acoustic = np.sort(positive)[:3]
            q_norms.append(q_si)
            branch_freqs.append(acoustic)
        q_norms = np.array(q_norms)
        branch_freqs = np.array(branch_freqs)  # (npts, 3) sorted ascending per point
        labels = ["T", "T2", "L"] if direction == "[110]" else ["T", "T", "L"]
        for column, label in enumerate(labels):
            omega_si = branch_freqs[:, column] * C_LIGHT_CM_S * 2.0 * np.pi  # rad/s
            velocity = float(np.sum(omega_si * q_norms) / np.sum(q_norms**2))
            key = (direction, label)
            lit = lit_v.get(key)
            rows.append(
                {
                    "material": mat,
                    "functional": functional,
                    "direction": direction,
                    "branch": label + ("A" if label in ("T", "L") else "A(001)"),
                    "column": column,
                    "v_calc_m_s": round(velocity, 1),
                    "v_lit_m_s": round(float(lit), 1) if lit else "",
                    "deviation_pct": round(100.0 * (velocity - lit) / lit, 2) if lit else "",
                }
            )
    return rows


def main() -> None:
    all_rows = []
    for mat in ("SrTiO3", "BaTiO3"):
        for functional in ("pbe", "pbesol"):
            all_rows.extend(acoustic_velocities(mat, functional))
    out = REPO / "data" / "processed" / "sound_velocities.csv"
    header = [
        "# sound_velocities.csv - produced by scripts/sound_velocities.py",
        "# calc: acoustic slopes of harmonic_dispersion_<mat>.csv fitted through the",
        "#   origin over the four smallest |q| along each direction",
        "# densities used for literature velocities and citations:",
    ]
    for mat, lit in LITERATURE.items():
        mass = CELL_MASS_AMU[mat] * AMU
        for functional in ("pbe", "pbesol"):
            try:
                volume = relaxed_lattice_m(mat, functional) ** 3
            except FileNotFoundError:
                continue
            header.append(
                f"#   {mat} [{functional}]: rho_calc = {mass / volume:.1f} kg/m^3 "
                f"(relaxed cell), rho_lit = {lit['density']:.0f} kg/m^3; {lit['citation']}"
            )
    frame = pd.DataFrame(all_rows).drop(columns=["column"])
    out.write_text("\n".join(header) + "\n" + frame.to_csv(index=False))
    print(frame.to_string(index=False))
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
