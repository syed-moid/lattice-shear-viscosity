#!/usr/bin/env python3
"""Basis sensitivity of construction A (maximum-overlap scalar transfer) inside exactly degenerate multiplets.

Production model (hybrid model, correction mesh 2^3 / inner mesh 12^3, 300 K, outer mesh 11^3). Every dynamical matrix that
is diagonalised receives a Hermitian perturbation of relative size 1e-10 (random, fixed seed), which leaves the spectrum
unchanged to round-off but picks an arbitrary basis inside each exactly degenerate eigenspace. Construction A takes the
coupling of the individual harmonic partner of largest overlap, so it depends on that basis; B and C are basis independent
inside exact multiplets and change only through the partner-based linewidth class.

Output: data/processed/v3_diagnostics/partner_basis_sensitivity.csv
Usage: uv run python scripts/v3_diagnostics/partner_basis_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG  # noqa: E402  (sets sys.path)

import compute_eta_SrTiO3 as E  # noqa: E402

T_K, EPS, SEEDS = 300, 1e-10, (0, 1, 2, 3, 4, 5)


def main():
    diagonalise = E.diagonalise
    surface = E.load_construction_surface(T_K)
    rows = []
    for seed in (None,) + SEEDS:
        rng = np.random.default_rng(seed)

        def perturbed(D, rng=rng, seed=seed):
            if seed is not None:
                N = rng.normal(size=D.shape) + 1j * rng.normal(size=D.shape)
                D = D + EPS * np.abs(D).max() * (N + N.conj().T) / 2
            return diagonalise(D)

        E.diagonalise = perturbed
        modes = E.construction_modes(T_K, surface=surface)
        eta = {c: float(E.assemble_construction(T_K, construction=c, modes=modes, surface=surface)[0]) for c in "ABC"}
        rows.append({"seed": "production" if seed is None else seed, **{f"eta_{c}_Pas": v for c, v in eta.items()}})
        print(rows[-1], flush=True)
    E.diagonalise = diagonalise
    df = pd.DataFrame(rows)
    for c in "ABC":
        df[f"eta_{c}_rel_pct"] = 100 * (df[f"eta_{c}_Pas"] / df[f"eta_{c}_Pas"].iloc[0] - 1)
    out = DIAG / "partner_basis_sensitivity.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# partner_basis_sensitivity.py: hybrid model 2^3/12^3, 300 K, 11^3; random Hermitian perturbation 1e-10 of every\n"
                 "# diagonalised dynamical matrix (seeds 0-5) selects an arbitrary basis inside exact multiplets; rel_pct relative to\n"
                 "# the unperturbed production evaluation.\n")
        df.to_csv(fh, index=False, float_format="%.10g")
    tmp.rename(out)
    s = df.iloc[1:]
    print(f"eta_A: {s.eta_A_rel_pct.min():+.2f} % to {s.eta_A_rel_pct.max():+.2f} %; eta_B {s.eta_B_rel_pct.min():+.3f} to "
          f"{s.eta_B_rel_pct.max():+.3f} %; eta_C {s.eta_C_rel_pct.min():+.3f} to {s.eta_C_rel_pct.max():+.3f} %")


if __name__ == "__main__":
    main()
