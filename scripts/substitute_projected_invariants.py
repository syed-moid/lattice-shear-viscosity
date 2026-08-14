#!/usr/bin/env python3
"""Substitution test for the degeneracy audit (SrTiO3, 300 K).

For every bare-degenerate multiplet audited in
scripts/audit_degeneracy_top20.py (the projected-subspace gauge check),
replace the tracked-assignment strain derivatives D = d(omega^2)/d(eps)
of the multiplet members by the eigenvalues of the strain perturbation
projected onto the bare-degenerate subspace (matched by sorted order),
and recompute the total eta_xyxy(300 K). The difference measures how
much the residual disagreement between the two constructions --- the
finite-strain curvature quantified independently in section 3.3 ---
propagates into the observable.

Reads : same inputs as compute_eta_SrTiO3.py + strained-cell mode files
Writes: data/processed/reports/substitution_test_projected_invariants.csv

Usage: uv run python scripts/substitute_projected_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_degeneracy_top20 import (  # noqa: E402
    BARE_DEGEN_TOL, TOL_MAIN, gauge_check, multiplet_of)
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
from compute_eta_SrTiO3 import REPO, assemble, load_vogt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.qe_modes import read_modes  # noqa: E402

T_K = 300


def main() -> None:
    masses = MASSES["SrTiO3"]
    directory = MODES_DIR / "SrTiO3"
    rows = compute_dataset(directory, masses, mesh_n=11)
    vogt, _ = load_vogt()
    eta0, _, _, details = assemble(T_K, rows, vogt, return_details=True)

    reference = read_modes(directory / "reference.modes")
    plus = read_modes(directory / "shear_xy_p005.modes")
    minus = read_modes(directory / "shear_xy_m005.modes")
    d_tracked = {(r["iq"], r["branch"]): r["D"] for r in rows}
    bare_omega = {(r["iq"], r["branch"]): r["omega_ref"] for r in rows}

    by_iq: dict[int, list[dict]] = {}
    for d in details:
        by_iq.setdefault(d["iq"], []).append(d)
    top = sorted(details, key=lambda d: abs(d["gruneisen"]), reverse=True)[:20]

    # identical multiplet selection to the audit
    multiplets = {}
    for d in top:
        iq = d["iq"]
        mult = multiplet_of(by_iq[iq], d, TOL_MAIN)
        branches = tuple(sorted(m["branch"] for m in mult))
        if len(branches) < 2:
            continue
        bare = np.array([bare_omega[(iq, b)] for b in branches])
        if (bare.max() - bare.min()) >= BARE_DEGEN_TOL:
            continue
        multiplets[(iq, branches)] = None

    lines = ["multiplet_q_index,branches,D_tracked_sorted,block_eigvals_sorted"]
    substituted = {}
    for (iq, branches) in sorted(multiplets):
        _, eigvals = gauge_check(iq, list(branches), reference, plus, minus,
                                 masses)
        tracked = np.array([d_tracked[(iq, b)] for b in branches])
        order_t = np.argsort(tracked)
        eig_sorted = np.sort(eigvals)
        for pos, b in enumerate(np.array(branches)[order_t]):
            substituted[(iq, int(b))] = float(eig_sorted[pos])
        lines.append(f"{iq},\"{';'.join(map(str, branches))}\","
                     f"\"{';'.join(f'{v:.1f}' for v in np.sort(tracked))}\","
                     f"\"{';'.join(f'{v:.1f}' for v in eig_sorted)}\"")

    rows_sub = []
    n_replaced = 0
    for r in rows:
        key = (r["iq"], r["branch"])
        if key in substituted:
            r = dict(r)
            r["D"] = substituted[key]
            n_replaced += 1
        rows_sub.append(r)

    eta_sub, _, _ = assemble(T_K, rows_sub, vogt)
    delta = eta_sub / eta0 - 1.0
    print(f"audited multiplets substituted: {len(multiplets)} "
          f"({n_replaced} mode-slots)")
    print(f"eta_xyxy(300 K): tracked = {eta0:.6e}, substituted = "
          f"{eta_sub:.6e}, Delta = {100.0 * delta:+.3f}%")

    rep = REPO / "data" / "processed" / "reports"
    header = [
        "# substitution_test_projected_invariants.csv -",
        "# scripts/substitute_projected_invariants.py",
        f"# Tracked-assignment D replaced by projected-subspace block eigenvalues",
        f"# for the {len(multiplets)} audited bare-degenerate multiplets "
        f"({n_replaced} mode-slots).",
        f"# eta_xyxy(300 K): tracked {eta0:.6e} Pa s, substituted "
        f"{eta_sub:.6e} Pa s, Delta = {100.0 * delta:+.3f}%.",
    ]
    out = rep / "substitution_test_projected_invariants.csv"
    out.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
