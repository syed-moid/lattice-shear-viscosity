#!/usr/bin/env python3
"""Machine-readable reconstruction ledger of eta_xyxy(SrTiO3, 300 K) (Supplement Table S8), full precision.

Each step records the value of a stated calculation and the file or record it comes from. Steps 0-9 are recorded outputs
of earlier calculations (submitted manuscript; tag v3.0 of this repository). Steps 10 and 11 are recomputed here on the
production model (hybrid model, correction mesh 2x2x2 / inner mesh 12x12x12, 11^3 outer mesh, construction C, 300 K):
  10: current coupling rule and degenerate linewidth averaging, harmonic strained sets with the QE rigid-ion term of
      each strained cell (HARMONIC_CONVENTION = "qe", the evaluation used before step 11);
  11: the same with the harmonic strained sets in the ALAMODE convention (production).
Ratios are to the previous step, from unrounded values.

Writes: data/processed/reconstruction_ledger_SrTiO3.csv
Usage: uv run python scripts/export_reconstruction_ledger.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
CC = "data/processed/coupling_constructions_SrTiO3.csv"
SENS = "data/processed/table_sensitivity_SrTiO3.csv"

RECORDED = [
    ("0", "submitted calculation (symmetric-path derivative, energy kernel, per-q rank pairing, example surface with the "
          "distributed precomputed corrections)", 3.887e-3, "submitted manuscript (recorded value)"),
    ("1-3", "engineering-shear derivative; stress-correlator kernel; exact isotope projection (isotope channel only)",
     9.721807e-4, f"tag v3.0, {SENS}, row 'legacy: tutorial surface, rank map'"),
    ("4", "per-q rank pairing -> eigenvector matching (same surface)", 1.148259e-3,
     f"tag v3.0, {SENS}, row 'legacy: tutorial surface, direct matching, transferred bare coupling'"),
    ("5", "own SCPH solution (diagnostic surface, 2^3/2^3), own linewidths; construction A", 1.91446e-3,
     f"tag v3.0, {CC}, row diagnostic 2/2 consistent, eta_A"),
    ("6", "A -> projected harmonic coupling B (same surface)", 8.48154e-4, f"tag v3.0, {CC}, row diagnostic 2/2 consistent, eta_B"),
    ("7", "B -> projected SCPH coupling C (same surface)", 5.84226e-4, f"tag v3.0, {CC}, row diagnostic 2/2 consistent, eta_C"),
    ("8", "input model: diagnostic surface -> hybrid model (2^3/2^3)", 5.11106e-4, f"tag v3.0, {CC}, row hybrid 2/2 consistent, eta_C"),
    ("9", "inner mesh 2^3 -> 12^3 (v3.0 reference)", 5.07583e-4, f"tag v3.0, {CC}, row hybrid 2/12 consistent, eta_C"),
]


def production_eta(convention: str) -> float:
    eta_mod.HARMONIC_CONVENTION = convention
    surface = eta_mod.load_construction_surface(T_K)
    modes = eta_mod.construction_modes(T_K, surface=surface)
    return float(eta_mod.assemble_construction(T_K, modes=modes, surface=surface)[0])


def main() -> None:
    rows = [{"step": s, "calculation": c, "eta_300K_Pas": v, "record": r} for s, c, v, r in RECORDED]
    rows.append({"step": "10", "calculation": "diagonal projection for split modes and degenerate linewidth averaging "
                 "(harmonic strained sets with the QE rigid-ion term)", "eta_300K_Pas": production_eta("qe"),
                 "record": "scripts/export_reconstruction_ledger.py, production model with HARMONIC_CONVENTION = 'qe'"})
    rows.append({"step": "11", "calculation": "translationally invariant (ALAMODE) convention of the harmonic strained sets "
                 "(reference)", "eta_300K_Pas": production_eta("alamode"),
                 "record": f"{CC}, row hybrid 2/12 consistent, eta_C; data/processed/eta_SrTiO3.csv, T = 300 K"})
    df = pd.DataFrame(rows)
    df["ratio_to_previous"] = df.eta_300K_Pas / df.eta_300K_Pas.shift(1)
    out = REPO / "data" / "processed" / "reconstruction_ledger_SrTiO3.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# export_reconstruction_ledger.py: eta_xyxy(300 K) along the reconstruction from the submitted to the present\n"
                 "# calculation (Supplement Table S8); Pa s, outer mesh 11^3; ratio to the previous step from unrounded values.\n"
                 "# Successive changes along this specified sequence; not independent error estimates.\n")
        df.to_csv(fh, index=False, float_format="%.10g")
    tmp.rename(out)
    print(df[["step", "eta_300K_Pas", "ratio_to_previous"]].to_string(index=False))
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
