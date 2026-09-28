#!/usr/bin/env python3
"""Scattering-mesh sensitivity of the production linewidths (hybrid model, 300 K, correction mesh 2^3 / inner mesh 12^3).

anphon MODE = RTA on the production renormalised set with the q mesh 8^3 (production), 10^3 and 12^3 (same deck otherwise;
TRISYM = 1 as in production), and the bare RTA of the same harmonic set on the same meshes for the character split of the
production frequency-class map. Linewidths are mapped to the 11^3 production modes by the production rule; frequencies,
eigenvectors and couplings (construction C) are the frozen ones. Reports eta_C(300 K) and kappa(300 K) per mesh.

Output: data/processed/v3_diagnostics/scattering_mesh_sensitivity.csv
Usage: uv run python scripts/v3_diagnostics/scattering_mesh_sensitivity.py
Needs: data/raw/alamode_sto/z_tut/i2s12, .../bare and .../i2s12_rta_mesh (raw archive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG, RAW  # noqa: E402  (sets sys.path)

import compute_eta_SrTiO3 as E  # noqa: E402

T_K = 300
MESHDIR = RAW / "z_tut" / "i2s12_rta_mesh"


def kappa(kl_path):
    for line in Path(kl_path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            v = [float(x) for x in line.split()]
            return (v[1] + v[5] + v[9]) / 3.0
    raise ValueError(kl_path)


def main():
    vogt, _ = E.load_vogt()
    base = E.load_construction_surface(T_K)
    modes = E.construction_modes(T_K, 11, base)
    variants = {8: (E.own_od1_rta(T_K), E.PROD["rta_bare"]),
                10: (MESHDIR / "STO_RTA_i2s12_od1_300K_k10.result", MESHDIR / "STO_RTA_z_bare_k10.result"),
                12: (MESHDIR / "STO_RTA_i2s12_od1_300K_k12.result", MESHDIR / "STO_RTA_z_bare_k12.result")}
    rows = []
    for n, (rta, rta_bare) in variants.items():
        spec = dict(E.PROD, rta=lambda T, p=rta: p, rta_bare=rta_bare)
        surf = E.load_construction_surface(T_K, spec=spec)
        eta, sectors, _ = E.assemble_construction(T_K, modes=modes, surface=surf, vogt=vogt)
        rows.append({"scattering_mesh": f"{n}^3", "eta_C_Pas": eta, "kappa_W_mK": kappa(Path(str(rta)).with_suffix(".kl")),
                     "soft_manifold_median_gamma_cm1": surf["soft_gamma_median"], "rta_result": Path(rta).name})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["eta_rel_to_8"] = df["eta_C_Pas"] / df["eta_C_Pas"].iloc[0] - 1.0
    df["kappa_rel_to_8"] = df["kappa_W_mK"] / df["kappa_W_mK"].iloc[0] - 1.0
    out = DIAG / "scattering_mesh_sensitivity.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# scattering_mesh_sensitivity.py: hybrid model, 300 K, 2^3/12^3; RTA q mesh 8^3 (production), 10^3, 12^3; linewidths\n"
                 "# mapped by the production frequency-class rule (bare RTA of the same mesh for the character split); construction C,\n"
                 "# frozen frequencies and couplings; sensitivity row only.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
