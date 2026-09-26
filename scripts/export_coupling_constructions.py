#!/usr/bin/env python3
"""Coupling-construction comparison for SrTiO3 (the data behind Fig. 4 and Tables 3-4).

Three ways of evaluating the shear-strain coupling Lambda of a renormalised mode, on the same surface:
  A  transferred coupling: the bare (harmonic) coupling of the bare mode of largest overlap (retired);
  B  projected harmonic coupling: e_r^dagger K_HA e_r with K_HA the harmonic strain derivative;
  C  projected SCPH coupling: e_r^dagger K_SCPH e_r with K_SCPH the strain derivative of the SCPH dynamical
     matrix (production).
Two input models: the hybrid model (production; example harmonic set + QE strain perturbation) and the
diagnostic surface (own QE-PBEsol harmonic set). Inner meshes (KMESH_SCPH) 2^3, 4^3, 8^3, 12^3 at correction
mesh (KMESH_INTERPOLATE) 2^3; linewidths either consistent (own 8^3 RTA of each surface) or held at the
2^3/2^3 values of the same input model (isolates the coupling-and-frequency side; frequencies and
occupation factors still vary). 300 K, 11^3 outer mesh, stress-correlator kernel, Vogt Gamma exception.

Writes data/processed/coupling_constructions_SrTiO3.csv   (eta_A, eta_B, eta_C per surface / mesh / linewidth rule)
       data/processed/coupling_ratio_bins_SrTiO3.csv      (per partner-omega_0 bin at inner mesh 12^3, both surfaces)
       data/processed/acoustic_limit_SrTiO3.csv           (Lambda_A/B/C and Lambda/q^2 of the acoustic branches)
Usage: uv run python scripts/export_coupling_constructions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
from latvisc.coupling import acoustic_character  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
Z = eta_mod.Z_DIR
OWN = eta_mod.ALAMODE_DIR / "own_od1"
MESHES = ["i2s2", "i2s4", "i2s8", "i2s12"]
BINS = [("imaginary", -np.inf, 0.0), ("[0-25)", 0, 25), ("[25-50)", 25, 50), ("[50-100)", 50, 100),
        ("[100-150)", 100, 150), ("[150-175)", 150, 175), ("[175-300)", 175, 300), ("[300-inf)", 300, np.inf)]


def spec(surface: str, mesh: str, fixed_lw: bool) -> dict:
    if surface == "hybrid":
        d = Z / mesh
        return {"dir": d, "xml": f"renorm_z_{{tag}}_{mesh}_od1_{{T}}K.xml",
                "rta": ("../i2s2/STO_RTA_i2s2_od1_{T}K.result" if fixed_lw else f"STO_RTA_{mesh}_od1_{{T}}K.result"),
                "fc": eta_mod.Z_FC, "rta_bare": eta_mod.TUT_Z_OD1["rta_bare"]}
    d = OWN / mesh
    return {"dir": d, "xml": f"renorm_{{tag}}_{mesh}_od1_{{T}}K.xml",
            "rta": ("../i2s2/STO_RTA_i2s2_od1_{T}K.result" if fixed_lw else f"STO_RTA_{mesh}_od1_{{T}}K.result"),
            "fc": eta_mod.QE_FC, "rta_bare": eta_mod.OWN_OD1["rta_bare"]}


def bin_of(w):
    for name, lo, hi in BINS:
        if lo <= w < hi:
            return name
    return "[300-inf)"


def write(df, name, header):
    out = REPO / "data" / "processed" / name
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("".join(f"# {h}\n" for h in header))
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    rows, bin_rows, ac_rows = [], [], []
    for surface in ("hybrid", "diagnostic"):
        for mesh in MESHES:
            for fixed in (False, True):
                if fixed and mesh == "i2s2":
                    continue
                sp = spec(surface, mesh, fixed)
                if not eta_mod.surface_available(T_K, sp):
                    print(f"  {surface} {mesh}: missing, skipped")
                    continue
                surf = eta_mod.load_construction_surface(T_K, spec=sp)
                modes = eta_mod.construction_modes(T_K, 11, surf)
                rec = {"surface": surface, "correction_mesh": 2, "inner_mesh": int(mesh.split("s")[1]),
                       "linewidths": "fixed_at_2x2x2" if fixed else "consistent"}
                det = {}
                for cons in ("A", "B", "C"):
                    e, _, _, d = eta_mod.assemble_construction(T_K, construction=cons, modes=modes, surface=surf, vogt=vogt,
                                                               return_details=True)
                    rec[f"eta_{cons}_Pas"] = e
                    det[cons] = d
                rows.append(rec)
                print(f"  {surface:10s} 2/{rec['inner_mesh']:<2d} {rec['linewidths']:15s} A {rec['eta_A_Pas']:.4e}  B {rec['eta_B_Pas']:.4e}  C {rec['eta_C_Pas']:.4e}", flush=True)
                if mesh == "i2s12" and not fixed:
                    acc = {}
                    for cons in ("A", "B", "C"):
                        for d in det[cons]:
                            b = bin_of(d["omega0"]) if d["sector"] != "gamma_sector" else "gamma_triplet"
                            acc.setdefault(b, {"n": 0, "A": 0.0, "B": 0.0, "C": 0.0})
                            acc[b][cons] += d["eta_contrib"]
                            if cons == "A":
                                acc[b]["n"] += 1
                    for b, v in acc.items():
                        bin_rows.append({"surface": surface, "inner_mesh": 12, "partner_omega0_bin": b, "n_modes": v["n"],
                                         "eta_A_Pas": v["A"], "eta_B_Pas": v["B"], "eta_C_Pas": v["C"],
                                         "ratio_B_over_A": v["B"] / v["A"] if v["A"] else np.nan,
                                         "ratio_C_over_B": v["C"] / v["B"] if v["B"] else np.nan})
                    # acoustic limit along Gamma-X and Gamma-M
                    qs, meta = [], []
                    for line, dvec in (("GX", (1, 0, 0)), ("GM", (1, 1, 0))):
                        dv = np.array(dvec, float) / np.linalg.norm(dvec)
                        for s in (0.025, 0.05, 0.075, 0.1, 0.15):
                            qs.append(tuple(s * dv)); meta.append((line, s, dv))
                    am = eta_mod.construction_modes(T_K, 11, surf, qpoints=qs)
                    masses = eta_mod.qe_set("reference", sp).masses_amu
                    for k, (line, s, dv) in enumerate(meta):
                        mk = [m for m in am if m["iq"] == k][:3]
                        omega_r, E_r = eta_mod.diagonalise(surf["sets"]["reference"].dynmat(qs[k], asr_onsite=True))
                        labels, _, _ = acoustic_character(E_r, masses, dv)
                        for j, m in enumerate(mk):
                            ac_rows.append({"surface": surface, "line": line, "q_reduced": s, "mode": j + 1, "character": labels[j],
                                            "omega_r_cm1": m["omega_r"], "Lambda_A_cm2": m["lam_A"], "Lambda_B_cm2": m["lam_B"],
                                            "Lambda_C_cm2": m["lam_C"], "Lambda_A_over_q2": m["lam_A"] / s ** 2,
                                            "Lambda_B_over_q2": m["lam_B"] / s ** 2, "Lambda_C_over_q2": m["lam_C"] / s ** 2})
    hdr = ["produced by scripts/export_coupling_constructions.py; SrTiO3, 300 K, 11^3 outer mesh, stress-correlator kernel,",
           "Vogt Gamma exception; A = transferred coupling (retired), B = projected harmonic coupling, C = projected SCPH coupling;",
           "surface hybrid = example harmonic set + QE strain perturbation (production); diagnostic = own QE-PBEsol harmonic set;",
           "correction mesh (KMESH_INTERPOLATE) 2^3, inner mesh (KMESH_SCPH) as listed; linewidths consistent (own RTA of each",
           "surface) or fixed at the 2^3/2^3 values of the same input model. Finite-mesh model results; no converged limit implied."]
    write(pd.DataFrame(rows), "coupling_constructions_SrTiO3.csv", hdr)
    write(pd.DataFrame(bin_rows), "coupling_ratio_bins_SrTiO3.csv",
          hdr[:3] + ["per bare-partner omega_0 bin (cm^-1) at inner mesh 12^3, consistent linewidths; ratio = eta ratio of the bin."])
    write(pd.DataFrame(ac_rows), "acoustic_limit_SrTiO3.csv",
          hdr[:3] + ["three lowest renormalised modes along Gamma-X and Gamma-M at inner mesh 12^3; q in reduced units (2 pi/a);",
                     "Lambda in cm^-2 per unit engineering shear; an acoustic strain coupling must scale as q^2 (Lambda/q^2 finite)."])


if __name__ == "__main__":
    main()
