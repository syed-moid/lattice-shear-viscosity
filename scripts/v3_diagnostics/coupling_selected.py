#!/usr/bin/env python3
"""coupling_selected.py - Steps 6.1b, 6.3b, 6.3c: the controlled comparison at selected q.

Reads  data/processed/v3_diagnostics/selection_6p3a.csv, data/processed/v3_diagnostics/mode_table_SrTiO3_300K_v4.csv, the QE .fc sets and the
       own-surface SCPH renormalised XMLs (data/raw/alamode_sto/own_surface_6p2, od0 and od1).
Writes data/processed/v3_diagnostics/coupling_constructions_selected.csv   (one row per selected production mode slot)
       data/processed/v3_diagnostics/coupling_constructions_allmodes.csv   (one row per renormalised mode at each selected q)
       data/processed/v3_diagnostics/acoustic_limit_6p3c.csv               (Gamma->X/M/R lines)
       data/processed/v3_diagnostics/K_matrices_selected.npz               (Hermitian K_HA, K_SCPH per q, both steps)
Units: cm^-2 per unit engineering shear h (= tensor eps_xy derivative, H3); gamma dimensionless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from coupling_kernel import (  # noqa: E402
    DEGEN_TOL, H_STEP, VARIANTS, Sets, acoustic_character, evaluate_q, gruneisen,
)

SEL = pd.read_csv(DIAG / "selection_6p3a.csv", comment="#")
TABLE = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v4.csv", comment="#")
ETA_TOTAL = float(TABLE.eta_contrib_Pas.sum())
LINES = {"GX": (1, 0, 0), "GM": (1, 1, 0), "GR": (1, 1, 1)}
QMAG = (0.05, 0.1, 0.2, 0.3)


def qkey(q):
    return tuple(np.round(q, 6))


def main():
    sel_q = {}
    for r in SEL.itertuples():
        sel_q.setdefault(qkey((r.qa, r.qb, r.qc)), int(r.iq))
    print(f"{len(sel_q)} distinct selected q")
    line_q = {}
    for name, d in LINES.items():
        d = np.array(d, dtype=float) / np.linalg.norm(d)
        for s in QMAG:
            line_q[(name, s)] = s * d

    all_rows, sel_rows, line_rows = [], [], []
    kstore = {}
    slot = TABLE.set_index(["iq", "branch"])
    for variant in VARIANTS:
        sets = Sets(variant)
        print(f"[{variant}] sets loaded")
        for q, iq in sel_q.items():
            res = evaluate_q(sets, q)
            for s in ("005", "010"):
                kstore[f"K_HA_{s}_{variant}_iq{iq}"] = sets.K_HA(q, s)
                kstore[f"K_SCPH_{s}_{variant}_iq{iq}"] = sets.K_SCPH(q, s)
            wr, wb = res["omega_r"], res["omega_b"]
            prod_index = res["A"]["prod_index"]
            # production eta weight per renormalised mode: sum of the v4 weights of the bare slots mapped to nu
            eta_nu = np.zeros(15)
            for b in range(15):
                eta_nu[prod_index[b]] += float(slot.loc[(iq, b + 1), "eta_contrib_Pas"])
            for nu in range(15):
                mu = int(res["A"]["mu_star"][nu])
                row = {"variant": variant, "iq": iq, "qa": q[0], "qb": q[1], "qc": q[2], "nu": nu + 1,
                       "omega_r_cm1": wr[nu], "multiplet_size": int(res["B"]["005"]["gsize"][nu]),
                       "bare_partner_mu": mu + 1, "omega_b_partner_cm1": wb[mu],
                       "overlap_single": res["A"]["overlap_single"][nu], "overlap_multiplet": res["A"]["overlap_summed"][nu],
                       "eta_weight_prod_Pas": eta_nu[nu],
                       "Lambda_A": res["A"]["lam"][nu],
                       "Lambda_bare_partner_table": float(slot.loc[(iq, mu + 1), "D_tensor_cm2"]) if iq is not None else np.nan,
                       "Lambda_B_diag": res["B"]["005"]["diag"][nu], "Lambda_B_eig": res["B"]["005"]["eig"][nu],
                       "TrK2_B": res["B"]["005"]["trk2"][nu], "offdiag_B_5cm1": res["B"]["005"]["offdiag"][nu],
                       "Lambda_B_h020": res["B"]["010"]["eig"][nu], "Lambda_B_richardson": res["B"]["rich"]["eig"][nu],
                       "Lambda_C_diag": res["C"]["005"]["diag"][nu], "Lambda_C_eig": res["C"]["005"]["eig"][nu],
                       "TrK2_C": res["C"]["005"]["trk2"][nu], "offdiag_C_5cm1": res["C"]["005"]["offdiag"][nu],
                       "Lambda_C_h020": res["C"]["010"]["eig"][nu], "Lambda_C_richardson": res["C"]["rich"]["eig"][nu],
                       "Lambda_C_noasr": res["C_noasr"]["eig"][nu],
                       "Lambda_C_matched_h010": res["C_matched"]["005"][nu], "Lambda_C_matched_h020": res["C_matched"]["010"][nu]}
                row["ratio_B_over_A"] = row["Lambda_B_eig"] / row["Lambda_A"] if row["Lambda_A"] != 0 else np.nan
                row["ratio_C_over_B"] = row["Lambda_C_eig"] / row["Lambda_B_eig"] if row["Lambda_B_eig"] != 0 else np.nan
                row["gamma_A"] = gruneisen(row["Lambda_A"], wr[nu])
                row["gamma_B"] = gruneisen(row["Lambda_B_eig"], wr[nu])
                row["gamma_C"] = gruneisen(row["Lambda_C_eig"], wr[nu])
                all_rows.append(row)
            # per selected slot
            for r in SEL[(np.isclose(SEL.qa, q[0])) & (np.isclose(SEL.qb, q[1])) & (np.isclose(SEL.qc, q[2]))].itertuples():
                b = int(r.branch) - 1
                nu = int(prod_index[b])
                a = all_rows[-15 + nu]
                sel_rows.append({"group": r.group, "variant": variant, "iq": iq, "qa": q[0], "qb": q[1], "qc": q[2],
                                 "branch": r.branch, "omega0_cm1": r.omega0_cm1, "omega_r_tutorial_cm1": r.omega_r_cm1,
                                 "sector": r.sector, "eta_contrib_prod_Pas": r.eta_contrib_Pas, "eta_share_pct": r.eta_share_pct,
                                 "D_tensor_table": float(slot.loc[(iq, r.branch), "D_tensor_cm2"]),
                                 "Lambda_bare_slot_projected": res["bare"]["005"]["eig"][b],
                                 "omega_b_builder_cm1": wb[b],
                                 "nu_own": nu + 1, "omega_r_own_cm1": wr[nu], "overlap_prod_map": res["A"]["prod_overlap"][b],
                                 "mu_star_of_nu": a["bare_partner_mu"], "mutual": int(a["bare_partner_mu"] == b + 1),
                                 **{k: a[k] for k in ("multiplet_size", "Lambda_A", "Lambda_B_eig", "Lambda_B_diag", "TrK2_B",
                                                      "offdiag_B_5cm1", "Lambda_B_h020", "Lambda_B_richardson", "Lambda_C_eig",
                                                      "Lambda_C_diag", "TrK2_C", "offdiag_C_5cm1", "Lambda_C_h020",
                                                      "Lambda_C_richardson", "Lambda_C_noasr", "Lambda_C_matched_h010",
                                                      "ratio_B_over_A", "ratio_C_over_B", "gamma_A", "gamma_B", "gamma_C")}})
        # 6.3c acoustic limit
        for (name, s), q in line_q.items():
            res = evaluate_q(sets, q)
            wr, Er = res["omega_r"], res["E_r"]
            labels, proj, frac = acoustic_character(Er, sets.masses, LINES[name])
            for nu in range(3):                       # three lowest renormalised modes
                line_rows.append({"variant": variant, "line": name, "q_mag": s, "qa": q[0], "qb": q[1], "qc": q[2],
                                  "nu": nu + 1, "character": labels[nu], "cm_projection_on_q": proj[nu],
                                  "cm_translation_fraction": frac[nu], "omega_r_cm1": wr[nu],
                                  "omega_b_partner_cm1": res["omega_b"][int(res["A"]["mu_star"][nu])],
                                  "overlap_single": res["A"]["overlap_single"][nu],
                                  "Lambda_A": res["A"]["lam"][nu], "Lambda_B": res["B"]["005"]["eig"][nu],
                                  "Lambda_C": res["C"]["005"]["eig"][nu], "Lambda_B_richardson": res["B"]["rich"]["eig"][nu],
                                  "Lambda_C_richardson": res["C"]["rich"]["eig"][nu],
                                  "Lambda_B_over_q2": res["B"]["005"]["eig"][nu] / s ** 2,
                                  "Lambda_C_over_q2": res["C"]["005"]["eig"][nu] / s ** 2,
                                  "Lambda_A_over_q2": res["A"]["lam"][nu] / s ** 2,
                                  "gamma_A": gruneisen(res["A"]["lam"][nu], wr[nu]),
                                  "gamma_B": gruneisen(res["B"]["005"]["eig"][nu], wr[nu]),
                                  "gamma_C": gruneisen(res["C"]["005"]["eig"][nu], wr[nu])})
            for s2 in ("005", "010"):
                kstore[f"K_HA_{s2}_{variant}_{name}_{s}"] = sets.K_HA(q, s2)
                kstore[f"K_SCPH_{s2}_{variant}_{name}_{s}"] = sets.K_SCPH(q, s2)
        print(f"[{variant}] done")

    hdr = ("# (coupling_selected.py); units cm^-2 per unit engineering shear h = 2s (H3: d/dh = d/d eps_xy);\n"
           "# own-surface unstrained renormalised basis (IFC supercell 2x2x2 quartic set, KMESH_INTERPOLATE 2x2x2,\n"
           "# KMESH_SCPH 2x2x2, 300 K), variant od0/od1 = SELF_OFFDIAG 0/1; *_eig = multiplet-block eigenvalue,\n"
           "# *_diag = e^dagger K e; step h = 0.010 unless _h020 (0.020) or _richardson; C_noasr = without the\n"
           "# on-site sum-rule projection of the renormalised sets; C_matched = eigenvalue-matched strained SCPH runs.\n")
    with open(DIAG / "coupling_constructions_allmodes.csv", "w") as fh:
        fh.write(hdr)
        pd.DataFrame(all_rows).to_csv(fh, index=False, float_format="%.6g")
    with open(DIAG / "coupling_constructions_selected.csv", "w") as fh:
        fh.write(hdr.replace("6.3b", "6.3b, one row per selected production slot (bare mode -> own-surface nu)"))
        pd.DataFrame(sel_rows).to_csv(fh, index=False, float_format="%.6g")
    with open(DIAG / "acoustic_limit_6p3c.csv", "w") as fh:
        fh.write("# (coupling_selected.py): three lowest own-surface renormalised modes along Gamma->X/M/R at\n"
                 "# |q| = 0.05, 0.1, 0.2, 0.3 (2 pi/a units); LA/TA from the centre-of-mass displacement direction;\n"
                 "# Lambda in cm^-2 per unit engineering shear (step h = 0.010); gamma = -Lambda/(2 omega_r^2).\n")
        pd.DataFrame(line_rows).to_csv(fh, index=False, float_format="%.6g")
    np.savez_compressed(DIAG / "K_matrices_selected.npz", **kstore)
    print("written")


if __name__ == "__main__":
    main()
