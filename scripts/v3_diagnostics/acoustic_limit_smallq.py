#!/usr/bin/env python3
"""Acoustic limit of the three couplings at small q on the production hybrid model (300 K, correction mesh 2^3, inner mesh
12^3; existing sets, no new SCPH).

Lines: Gamma-M (direction (1,1,0)/sqrt 2; transverse branches polarised along [1-10] and [001]), Gamma-R ((1,1,1)/sqrt 3)
and Gamma-X ((1,0,0), where every first-order xy-shear coupling vanishes by symmetry). |q| in reduced units (2 pi/a):
0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003. For each acoustic mode: Lambda_A (transferred from the maximum-overlap bare
partner), Lambda_B (projected harmonic), Lambda_C (projected SCPH), each from the raw perturbation K and from its symmetric
part K_sym (see strain_symmetry_check.py), Lambda/q^2, and for A the partner index, partner frequency and overlaps.

Output: data/processed/v3_diagnostics/acoustic_limit_smallq.csv
Usage: uv run python scripts/v3_diagnostics/acoustic_limit_smallq.py
Needs: data/raw/alamode_sto/z_tut (hybrid model, raw archive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG  # noqa: E402  (sets sys.path)

import strain_symmetry_check as S  # noqa: E402
from latvisc.coupling import (AlamodeSet, acoustic_character, diagonalise, project_coupling,  # noqa: E402
                              strain_derivative_matrix, transfer_bare_coupling)
from _paths import RAW  # noqa: E402

EXACT = S.E.EXACT_DEGEN_TOL          # direct, ungrouped projection for distinct branches; blocks only for exact degeneracy
FORMER = S.E.DEGEN_TOL               # former 0.5 cm^-1 grouping (for comparison)


def translation_residual(K, masses):
    """acoustic sum rule of K itself: K(q=0) applied to the three mass-weighted uniform translations (relative to ||K||)."""
    nat = len(masses)
    res = 0.0
    for a in range(3):
        t = np.zeros(3 * nat)
        t[a::3] = np.sqrt(masses)
        t /= np.linalg.norm(t)
        res = max(res, np.linalg.norm(K @ t))
    return res / np.linalg.norm(K)

Q_MAGS = (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003)
LINES = {"GM": (1, 1, 0), "GR": (1, 1, 1), "GX": (1, 0, 0)}


def main():
    group = S.shear_group()
    surf, sets, zb, Ks, ref = S.load_sets()
    s_bare = S.pick_sign(ref["bare_qe"], group, "bare, QE convention")[0]
    s_scph = S.pick_sign(ref["scph"], group, "scph")[0]
    # harmonic strained sets in the production (ALAMODE) convention, as the SCPH sets: the dipole term is subtracted and
    # restored in real space, so D(+-h) are Hermitian and K_HA is translationally invariant. The former evaluation (QE rigid-ion
    # term of each sheared cell, symmetrised) is kept as the comparison columns *_qeconv.
    Z = RAW / "z_tut"
    b0 = AlamodeSet(Z / "i2s12" / "z_reference_full_fc2.xml", S.E.Z_FC["reference"])
    bp = AlamodeSet(Z / "z_shear_xy_p005_full_fc2.xml", S.E.Z_FC["shear_xy_p005"])
    bm = AlamodeSet(Z / "z_shear_xy_m005_full_fc2.xml", S.E.Z_FC["shear_xy_m005"])
    bare_al = lambda q: b0.dynmat(q, asr_onsite=True)  # noqa: E731
    KHA = lambda q: strain_derivative_matrix(bp.dynmat(q, asr_onsite=True), bm.dynmat(q, asr_onsite=True), S.H)  # noqa: E731
    KHA_qe, KSC = Ks["K_HA (QE rigid-ion convention)"], Ks["K_SCPH (production, C)"]
    s_bare_al = S.pick_sign(bare_al, group, "bare, ALAMODE convention")[0]
    masses = zb["reference"].masses_amu
    rows = []
    for line, d in LINES.items():
        dv = np.array(d, float) / np.linalg.norm(d)
        for qm in Q_MAGS:
            q = qm * dv
            wb_al, Eb_al = diagonalise(bare_al(q))
            wb_qe, Eb_qe = diagonalise(ref["bare_qe"](q))
            wr, Er = diagonalise(ref["scph"](q))
            labels, proj, _ = acoustic_character(Er, masses, dv)
            K = {"raw": (KHA(q), KSC(q)), "sym": (S.symmetrise(KHA, q, group, s_bare_al), S.symmetrise(KSC, q, group, s_scph)),
                 "qeconv": (S.symmetrise(KHA_qe, q, group, s_bare), S.symmetrise(KSC, q, group, s_scph))}
            res = {}
            for tag, (kha, ksc) in K.items():
                wb, Eb = (wb_qe, Eb_qe) if tag == "qeconv" else (wb_al, Eb_al)
                pb = project_coupling(kha, Eb, wb, EXACT)
                lamA, mu, ov1, ovm = transfer_bare_coupling(pb["lam"], Eb, Er, wr, EXACT)
                res[tag] = {"A": lamA, "B": project_coupling(kha, Er, wr, EXACT)["lam"],
                            "C": project_coupling(ksc, Er, wr, EXACT)["lam"], "mu": mu, "ov1": ov1, "ovm": ovm}
                pbg = project_coupling(kha, Eb, wb, FORMER)
                res[tag]["A_grouped"] = transfer_bare_coupling(pbg["lam"], Eb, Er, wr, FORMER)[0]
                res[tag]["B_grouped"] = project_coupling(kha, Er, wr, FORMER)["lam"]
                res[tag]["C_grouped"] = project_coupling(ksc, Er, wr, FORMER)["lam"]
            for nu in range(3):
                pol = ""
                if line == "GM" and labels[nu] == "TA":
                    e = Er[:, nu].reshape(-1, 3)
                    u = (np.sqrt(masses)[:, None] * e).sum(axis=0)
                    pol = "[1-10]" if abs(np.vdot(np.array([1, -1, 0]) / np.sqrt(2), u)) ** 2 > 0.5 * np.vdot(u, u).real else "[001]"
                r = {"line": line, "q_reduced": qm, "mode": nu + 1, "character": labels[nu], "polarisation": pol,
                     "omega_r_cm1": wr[nu], "partner_bare_index": int(res["raw"]["mu"][nu]) + 1,
                     "partner_omega_bare_cm1": float(wb[res["raw"]["mu"][nu]]), "partner_overlap_single": res["raw"]["ov1"][nu],
                     "partner_overlap_multiplet": res["raw"]["ovm"][nu], "ta_splitting_cm1": float(abs(wr[1] - wr[0])),
                     "grouped_at_0p5": bool(abs(wr[1] - wr[0]) < FORMER and nu < 2)}
                for tag in ("raw", "sym", "qeconv"):
                    for c in ("A", "B", "C"):
                        r[f"Lambda_{c}_{tag}"] = res[tag][c][nu]
                        r[f"Lambda_{c}_{tag}_over_q2"] = res[tag][c][nu] / qm ** 2
                        r[f"Lambda_{c}_{tag}_grouped0p5_over_q2"] = res[tag][f"{c}_grouped"][nu] / qm ** 2
                rows.append(r)
            print(f"{line} {qm}: done", flush=True)
    q0 = np.zeros(3)
    qs = 1e-4 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)       # q -> 0 along Gamma-M (the non-analytic term is direction dependent)
    tr = {"K_HA": translation_residual(KHA(qs), masses), "K_HA_qeconv": translation_residual(KHA_qe(qs), masses),
          "K_SCPH": translation_residual(KSC(qs), masses)}
    print("translational invariance of K at q = 0 (||K t|| / ||K||):", {k: f"{v:.1e}" for k, v in tr.items()})
    df = pd.DataFrame(rows)
    for k, v in tr.items():
        df[f"translation_residual_{k}"] = v
    out = DIAG / "acoustic_limit_smallq.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# acoustic_limit_smallq.py: three lowest renormalised modes, hybrid model, 300 K, 2^3/12^3; |q| reduced (2 pi/a) along\n"
                 "# the unit direction of each line; Lambda in cm^-2 per unit engineering shear; A = maximum-overlap scalar transfer,\n"
                 "# B = projected harmonic, C = projected SCPH; _raw from K, _sym from its symmetric part K_sym; partner = bare mode of\n"
                 "# largest overlap (overlap_single) and the overlap summed over the renormalised multiplet (overlap_multiplet).\n"
                 "# Direct, ungrouped projections for distinct branches (blocks only for exact degeneracy, 1e-3 cm^-1); *_grouped0p5\n"
                 "# = former 0.5 cm^-1 grouping; ta_splitting_cm1 = splitting of the two lowest modes; translation_residual_* =\n"
                 "# ||K t|| / ||K|| at |q| = 1e-4 along Gamma-M for the mass-weighted uniform translations t (acoustic sum rule of K).\n"
                 "# K_HA: harmonic strained sets in the production ALAMODE convention (translationally invariant); *_qeconv: the former\n"
                 "# evaluation (QE-convention rigid-ion term of the strained cells, symmetrised; not translationally invariant).\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    cols = ["line", "q_reduced", "mode", "character", "polarisation", "ta_splitting_cm1", "grouped_at_0p5", "partner_overlap_single",
            "Lambda_A_sym_over_q2", "Lambda_B_sym_over_q2", "Lambda_C_sym_over_q2", "Lambda_A_sym_grouped0p5_over_q2",
            "Lambda_B_sym_grouped0p5_over_q2"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
