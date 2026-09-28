#!/usr/bin/env python3
"""Multiplets of the production 300 K modes: symmetry-enforced vs near-degenerate clusters, and the grouping sensitivity.

For every q of the 11^3 mesh the renormalised modes (hybrid model, 300 K, 2^3/12^3) are grouped with the production tolerance
(0.5 cm^-1). A group is a SYMMETRY-ENFORCED multiplet when its frequencies agree to 1e-3 cm^-1 AND its subspace is irreducible
under the little group G_q of O_h (sum_R |Tr(E_sub^+ U(R,q) E_sub)|^2 / |G_q| = 1, with the representation U of
strain_symmetry_check.py). Otherwise it is a near-degenerate cluster (grouped by the tolerance only). Singlets are listed
separately. The threshold 1e-3 cm^-1 sits in an empty gap of the spread distribution: SCPH round-off splits symmetry
multiplets by 1e-6 to 1e-4 cm^-1 (the renormalised dynamical matrix is symmetric to about 1e-8), no group has a spread
between 1e-4 and 1e-2 cm^-1, and every group above 1e-2 cm^-1 is reducible (accidental).

Linewidths: every RTA deck uses TRISYM = 1; the production linewidth of a mode is the frequency-class median of the RTA result
evaluated at the mode frequency, so modes of an exact multiplet receive identical linewidths; the raw per-mode RTA linewidths
inside degenerate sets of the 8^3 result are compared as well.

eta_C(300 K) is recomputed with grouping tolerance 0.05, 0.5 (production) and 2 cm^-1 (projected-block eigenvalues inside a
group; frequencies and linewidths per mode).

Outputs: data/processed/v3_diagnostics/multiplet_classes.csv and multiplet_classes_summary.txt
Usage: uv run python scripts/v3_diagnostics/multiplet_classes.py
Needs: data/raw/alamode_sto/z_tut (raw archive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG  # noqa: E402  (sets sys.path)

import compute_eta_SrTiO3 as E  # noqa: E402
import strain_symmetry_check as S  # noqa: E402
from latvisc.coupling import diagonalise, multiplet_groups  # noqa: E402
from crosscheck_alamode_sto_tau import parse_result  # noqa: E402

T_K = 300
EXACT_TOL = 1e-3
RTA_DEGEN_TOL = 1e-6


def little_group_oh(q):
    return [R for R in S.oh_operations() if np.allclose(((R @ q - q) + 0.5) % 1.0 - 0.5, 0.0, atol=1e-9)]


def main():
    surf = E.load_construction_surface(T_K)
    vogt, _ = E.load_vogt()
    sign = S.pick_sign(lambda q: surf["sets"]["reference"].dynmat(q, asr_onsite=True), S.shear_group(), "scph")[0]
    modes = E.construction_modes(T_K, 11, surf)          # production rule: blocks only inside exact degeneracies
    eta_prod, _, _, details = E.assemble_construction(T_K, modes=modes, surface=surf, vogt=vogt, return_details=True)
    contrib = {(d["iq"], d["nu"]): d["eta_contrib"] for d in details}
    qmesh = E.mesh_points(11)
    rows = []
    for iq, q in enumerate(qmesh):
        q = np.array(q)
        w, V = diagonalise(surf["sets"]["reference"].dynmat(q, asr_onsite=True))
        G = little_group_oh(q)
        for g in multiplet_groups(w, 0.5):
            spread = float(w[g].max() - w[g].min())
            if len(g) == 1:
                cls = "singlet"
                irr = 1.0
            else:
                Es = V[:, g]
                irr = sum(abs(np.trace(Es.conj().T @ S.u_matrix(R, q, sign) @ Es)) ** 2 for R in G) / len(G)
                if spread <= EXACT_TOL and abs(irr - 1.0) < 1e-3:
                    cls = "symmetry-enforced"
                elif spread <= EXACT_TOL:
                    cls = "degenerate, reducible (accidental or several irreps)"
                else:
                    cls = "near-degenerate cluster"
            rows.append({"iq": iq, "qa": q[0], "qb": q[1], "qc": q[2], "size": len(g), "omega_min_cm1": float(w[g].min()),
                         "spread_cm1": spread, "little_group_order": len(G), "irreducibility_index": float(irr), "class": cls,
                         "eta_contrib_Pas": float(sum(contrib.get((iq, i + 1), 0.0) for i in g))})
    df = pd.DataFrame(rows)
    share = df.groupby("class")["eta_contrib_Pas"].sum() / eta_prod
    counts = df.groupby("class").size()

    # raw RTA linewidths inside degenerate sets of the production 8^3 result (TRISYM = 1)
    freq, gam = parse_result(E.own_od1_rta(T_K), target_temp=T_K)
    by_q = {}
    for (qq, b), wv in freq.items():
        by_q.setdefault(qq, []).append((wv, gam.get((qq, b), np.nan)))
    rel_spreads = []
    for qq, lst in by_q.items():
        lst.sort()
        ws = np.array([a for a, _ in lst]); gs = np.array([b for _, b in lst])
        for g in multiplet_groups(ws, RTA_DEGEN_TOL):
            if len(g) > 1 and np.all(ws[g] > 1.0) and np.all(np.isfinite(gs[g])) and gs[g].mean() > 0:
                rel_spreads.append(float((gs[g].max() - gs[g].min()) / gs[g].mean()))
    rel_spreads = np.array(rel_spreads)

    # grouping tolerance sensitivity
    sens = []
    prod_tol = E.COUPLING_DEGEN_TOL
    for tol in (0.05, 0.5, 2.0):                         # former treatment: grouping of genuinely split modes (sensitivity)
        E.COUPLING_DEGEN_TOL = tol
        m = E.construction_modes(T_K, 11, surf)
        e, _, _ = E.assemble_construction(T_K, modes=m, surface=surf, vogt=vogt)
        sens.append((tol, e))
    E.COUPLING_DEGEN_TOL = prod_tol
    gap = df[(df['size'] > 1) & (df.spread_cm1 > 1e-4) & (df.spread_cm1 < 1e-2)]
    lines = [f"eta_C(300 K) production = {eta_prod:.6e} Pa s; groups on the 11^3 mesh with tolerance 0.5 cm^-1 "
             f"(exactness threshold {EXACT_TOL} cm^-1; groups with spread in (1e-4, 1e-2) cm^-1: {len(gap)}):"]
    for c in share.index:
        lines.append(f"  {c}: {counts[c]} groups, eta share {share[c]:.4%}")
    lines.append(f"raw 8^3 RTA linewidths (TRISYM = 1) inside degenerate sets (|d omega| <= 1e-6 cm^-1): {len(rel_spreads)} sets; "
                 f"relative spread max {rel_spreads.max():.2e}, median {np.median(rel_spreads):.2e}; production linewidths are "
                 f"averaged over exact degenerate sets before the frequency-class map (compute_eta_SrTiO3.average_degenerate_linewidths)")
    for tol, e in sens:
        lines.append(f"grouping tolerance {tol} cm^-1: eta_C = {e:.6e} Pa s ({100 * (e / eta_prod - 1):+.3f} % vs production)")
    out = DIAG / "multiplet_classes.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# multiplet_classes.py: groups of renormalised modes (tolerance 0.5 cm^-1), hybrid model, 300 K, 2^3/12^3, 11^3 mesh;\n"
                 "# class: symmetry-enforced (spread <= 1e-3 cm^-1 and irreducible under the little group of O_h) | near-degenerate cluster\n"
                 "# | degenerate, reducible | singlet; eta_contrib = summed production contribution of the group (construction C).\n")
        df.to_csv(fh, index=False, float_format="%.8g")
        for tol, e in sens:
            fh.write(f"# grouping_tolerance_cm1={tol},eta_C_Pas={e:.8e}\n")
    tmp.rename(out)
    txt = DIAG / "multiplet_classes_summary.txt"
    tmp = txt.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(txt)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
