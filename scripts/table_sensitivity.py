#!/usr/bin/env python3
"""Construction and model sensitivity table for eta_xyxy(SrTiO3, 300 K) (former Table 1 slot).

Rows (all at 300 K, 11^3 outer mesh):
  hybrid model (production; correction mesh 2x2x2 / inner mesh 12x12x12, SELF_OFFDIAG = 1): projected SCPH
    coupling C, projected harmonic coupling B, transferred coupling A;
  C with the coupling from the 0.010-amplitude pair (h = 0.020), Richardson (0.010/0.020) on the
    asymptotic modes only, Richardson everywhere (sensitivity);
  C with nearest-q linewidths instead of the frequency-class map;
  diagnostic surface (own QE-PBEsol harmonic set): C at 2x2x2/2x2x2, C at 2x2x2/12x12x12 with consistent
    linewidths and with the linewidths fixed at the 2x2x2/2x2x2 values; C with symmetrised strained
    harmonic inputs (even offset removed before SCPH); C at 4x4x4/4x4x4 and 4x4x4/8x8x8;
  legacy: tutorial surface, direct eigenvector matching, transferred bare coupling (an earlier
    model); tutorial surface, rank map (an earlier model).
Writes: data/processed/table_sensitivity_SrTiO3.csv
Usage: uv run python scripts/table_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
OWN = eta_mod.ALAMODE_DIR / "own_od1"
VARIANTS = {
    "diagnostic surface 2x2x2/2x2x2, C": {"dir": OWN / "i2s2", "xml": "renorm_{tag}_i2s2_od1_{T}K.xml", "rta": "STO_RTA_i2s2_od1_{T}K.result"},
    "diagnostic surface 2x2x2/12x12x12, C, consistent linewidths": {"dir": OWN / "i2s12", "xml": "renorm_{tag}_i2s12_od1_{T}K.xml",
                                                                  "rta": "STO_RTA_i2s12_od1_{T}K.result"},
    "diagnostic surface 2x2x2/12x12x12, C, linewidths fixed at 2x2x2/2x2x2": {"dir": OWN / "i2s12", "xml": "renorm_{tag}_i2s12_od1_{T}K.xml",
                                                                            "rta": "../i2s2/STO_RTA_i2s2_od1_{T}K.result"},
    "own 4x4x4/4x4x4, SELF_OFFDIAG=1, C": {"dir": OWN / "i4s4", "xml": "renorm_{tag}_i4s4_od1_{T}K.xml", "rta": "STO_RTA_i4s4_od1_{T}K.result"},
    "own 4x4x4/8x8x8, SELF_OFFDIAG=1, C": {"dir": OWN / "i4s8", "xml": "renorm_{tag}_i4s8_od1_{T}K.xml", "rta": "STO_RTA_i4s8_od1_{T}K.result"},
    "diagnostic surface 2x2x2/2x2x2, symmetrised strained inputs, C": {"dir": OWN / "i2s2_sym_eval", "xml": "renorm_own_{tag}_od1_{T}K.xml",
                                                        "rta": "../i2s2/STO_RTA_i2s2_od1_{T}K.result"},
}


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    surface = eta_mod.load_construction_surface(T_K, with_h020=True)
    modes = eta_mod.construction_modes(T_K, 11, surface)
    rows = []

    def add(label, eta, note=""):
        rows.append((label, eta, note))
        print(f"  {label:60s} {eta:.4e}  {note}")

    eta_c, _, _ = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt)
    add("hybrid model 2x2x2/12x12x12, projected SCPH coupling C (production)", eta_c, "h = 0.010; finite-mesh result")
    for cons in ("B", "A"):
        e, _, _ = eta_mod.assemble_construction(T_K, construction=cons, modes=modes, surface=surface, vogt=vogt)
        add(f"hybrid model 2x2x2/12x12x12, {'projected harmonic coupling B' if cons == 'B' else 'transferred coupling A'}", e,
            "same surface, same basis" if cons == "B" else "transferred bare coupling (retired)")
    if "lam_C_h020" in modes[0]:
        e, _, _ = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, coupling_key="lam_C_h020")
        add("C, coupling from the 0.010-amplitude pair (h = 0.020)", e, "outside the linear regime for the near-unstable manifold")
        asym = [m for m in modes if m["lam_C"] != 0 and abs(m["lam_C_h020"] / m["lam_C"] - 1.0) < 0.25
                and np.sign(m["lam_C_h020"]) == np.sign(m["lam_C"])]
        keys = {(m["iq"], m["nu"]) for m in asym}
        mixed = [dict(m, lam_C=(m["lam_C_rich"] if (m["iq"], m["nu"]) in keys else m["lam_C"])) for m in modes]
        e, _, _ = eta_mod.assemble_construction(T_K, modes=mixed, surface=surface, vogt=vogt)
        add("C, Richardson (0.010/0.020) on asymptotic modes only", e, f"{100 * len(asym) / len(modes):.0f} % of modes asymptotic")
        e, _, _ = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, coupling_key="lam_C_rich")
        add("C, Richardson everywhere", e, "sensitivity, not an extrapolation")
    e, _, _ = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, gamma_source="nearest_q")
    add("C, nearest-q linewidths (8^3 -> 11^3)", e, "instead of the frequency-class map")
    for label, spec in VARIANTS.items():
        try:
            s2 = eta_mod.load_construction_surface(T_K, spec=spec)
            m2 = eta_mod.construction_modes(T_K, 11, s2)
            e, _, _ = eta_mod.assemble_construction(T_K, modes=m2, surface=s2, vogt=vogt)
            add(label, e, "diagnostic surface (different input model)")
        except FileNotFoundError as exc:
            print(f"  {label}: skipped ({exc})")
    rows_legacy = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    e, _, _ = eta_mod.assemble(T_K, rows_legacy, vogt, surface_tag="tutorial", omega_r_source="direct")
    add("legacy: tutorial surface, direct matching, transferred bare coupling", e, "earlier model, direct matching")
    e, _, _ = eta_mod.assemble(T_K, rows_legacy, vogt, omega_r_source="rankmap")
    add("legacy: tutorial surface, rank map", e, "earlier model, rank map")
    out = REPO / "data" / "processed" / "table_sensitivity_SrTiO3.csv"
    header = [
        "# table_sensitivity_SrTiO3.csv - produced by scripts/table_sensitivity.py",
        "# construction and model sensitivity of eta_xyxy(300 K), 11^3 outer mesh; production row (hybrid model) first; legacy rows",
        "# are different surfaces and different models (ledger), not the same model with one switch changed.",
        "row,eta_300K_Pas,ratio_to_production,note",
    ]
    lines = [f"\"{lab}\",{e:.6e},{e / eta_c:.4f},\"{note}\"" for lab, e, note in rows]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
