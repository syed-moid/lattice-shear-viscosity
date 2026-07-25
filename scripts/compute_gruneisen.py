#!/usr/bin/env python3
"""Mode Grueneisen parameters from strained PBEsol phonon runs (A5').

For each material, reads matdyn mesh eigenvector files (.mesh11.modes)
for the unstrained reference and each strain, matches modes between
strained and reference calculations by mass-weighted eigenvector overlap
|<z_ref | z_strained>| (never by frequency ordering), and evaluates
central differences:

    gamma_xy(qs)  = - (omega[+eps] - omega[-eps]) / (2 * eps * omega_ref)
    gamma_vol(qs) = - (ln omega[+d] - ln omega[-d]) / (ln V+ - ln V-)

Degenerate reference subspaces (|domega| < DEGEN_TOL) are matched as a
block: strained frequencies are assigned to the subspace by total overlap
and sorted within it, giving the invariant combinations (eigenvalues of
the strain perturbation on the subspace).

Modes belonging to the unstable manifold (omega_ref < OMEGA_MIN) are
written with gamma = nan and flagged: harmonic gamma is undefined there
(Route H replaces these modes by renormalized frequencies).

Reads : data/raw/gruneisen_modes/<material>/{reference,<strain>}.modes
Writes: data/processed/gruneisen_<material>.csv

Usage: uv run python scripts/compute_gruneisen.py [SrTiO3] [BaTiO3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import match_modes_by_overlap  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODES_DIR = REPO / "data" / "raw" / "gruneisen_modes"

MASSES = {
    "SrTiO3": np.array([87.62, 47.867, 15.999, 15.999, 15.999]),
    "BaTiO3": np.array([137.327, 47.867, 15.999, 15.999, 15.999]),
}
SHEAR_PAIRS = {"eps05": ("shear_xy_p005", "shear_xy_m005", 0.005),
               "eps10": ("shear_xy_p010", "shear_xy_m010", 0.010)}
HYDRO = ("hydro_p005", "hydro_m005", 0.005)
DEGEN_TOL = 0.5   # cm^-1, reference subspace grouping
OMEGA_MIN = 5.0   # cm^-1, below this (incl. negative) gamma is not defined
ACOUSTIC_SKIP = 1e-3  # |q| below this: acoustic modes are translations


def gamma_from_pair(freq_ref, matched_plus, matched_minus, amplitude, volumetric):
    with np.errstate(divide="ignore", invalid="ignore"):
        if volumetric:
            dln = np.log(matched_plus) - np.log(matched_minus)
            dlnv = 3.0 * (np.log1p(amplitude) - np.log1p(-amplitude))
            return -dln / dlnv
        return -(matched_plus - matched_minus) / (2.0 * amplitude * freq_ref)


def process(material: str) -> None:
    masses = MASSES[material]
    directory = MODES_DIR / material
    reference = read_modes(directory / "reference.modes")
    strains = {
        name: read_modes(directory / f"{name}.modes")
        for name in [s for pair in SHEAR_PAIRS.values() for s in pair[:2]] + list(HYDRO[:2])
    }

    rows = ["q_index,qx,qy,qz,branch,omega_ref_cm1,"
            "gamma_xy_eps05,gamma_xy_eps10,gamma_vol,unstable_flag,min_overlap"]
    for iq, (q, freq_ref, vec_ref) in enumerate(reference):
        matched, quality = {}, []
        for name, blocks in strains.items():
            q_s, freq_s, vec_s = blocks[iq]
            # Printed q is matdyn's Cartesian conversion of the shared
            # fractional input using each cell's own (strain-distorted)
            # reciprocal lattice — it differs slightly by strain even for
            # identical fractional labels (see scripts/gen_matdyn_mesh_inputs.py
            # and the G1' incident log). Correspondence is by index/order,
            # not printed-value equality; this is a coarse misalignment check.
            assert np.allclose(q, q_s, atol=0.01), f"q mismatch at {iq} for {name}"
            m, overlap = match_modes_by_overlap(freq_ref, vec_ref, freq_s, vec_s, masses)
            matched[name] = m
            quality.append(overlap.max(axis=1).min())
        gxy05 = gamma_from_pair(freq_ref, matched["shear_xy_p005"],
                                matched["shear_xy_m005"], 0.005, False)
        gxy10 = gamma_from_pair(freq_ref, matched["shear_xy_p010"],
                                matched["shear_xy_m010"], 0.010, False)
        gvol = gamma_from_pair(freq_ref, matched["hydro_p005"],
                               matched["hydro_m005"], 0.005, True)
        acoustic_q = np.linalg.norm(q) < ACOUSTIC_SKIP
        for branch in range(len(freq_ref)):
            unstable = freq_ref[branch] < OMEGA_MIN
            skip = unstable or (acoustic_q and abs(freq_ref[branch]) < OMEGA_MIN)
            values = ["nan", "nan", "nan"] if skip else [
                f"{gxy05[branch]:.4f}", f"{gxy10[branch]:.4f}", f"{gvol[branch]:.4f}"]
            rows.append(
                f"{iq},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{branch + 1},"
                f"{freq_ref[branch]:.4f},{','.join(values)},"
                f"{int(unstable)},{min(quality):.3f}"
            )

    out = REPO / "data" / "processed" / f"gruneisen_{material}.csv"
    header = [
        f"# gruneisen_{material}.csv - produced by scripts/compute_gruneisen.py",
        "# PBEsol strained-cell central differences on the 11x11x11 mesh;",
        "# modes matched by mass-weighted eigenvector overlap, degenerate",
        "# subspaces block-matched and sorted (invariant combinations).",
        "# gamma undefined (nan) for the unstable manifold (omega_ref < 5 cm^-1)",
        "# and for acoustic translations at Gamma. min_overlap is the worst",
        "# best-match overlap across the six strains at that q (1.0 = clean).",
        "functional: pbesol",
    ]
    out.write_text("\n".join("# " + h if not h.startswith("#") else h for h in header)
                   + "\n" + "\n".join(rows) + "\n")
    print(f"{material}: {len(reference)} q-points -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    for material in sys.argv[1:] or ["SrTiO3", "BaTiO3"]:
        process(material)
