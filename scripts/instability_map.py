#!/usr/bin/env python3
"""Instability map: every harmonic mode with omega^2 < 0 along the q-path.

Tabulates imaginary modes at the high-symmetry points and the unstable
regions along Gamma-X-M-Gamma-R-X, with irrep assignments at the
high-symmetry points (standard perovskite assignments consistent with the
dynmat.x eigenvectors at Gamma and the branch connectivity).

Reads : data/processed/harmonic_dispersion_<material>.csv
Writes: data/processed/instability_map.csv

Usage: uv run python scripts/instability_map.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NODES = {0: "Gamma", 40: "X", 80: "M", 120: "Gamma", 160: "R", 200: "X"}
SEGMENTS = {0: "Gamma-X", 1: "X-M", 2: "M-Gamma", 3: "Gamma-R", 4: "R-X"}

# Irrep labels at high-symmetry points for the perovskite instabilities
# (Glazer/Cowley conventions): Gamma15 = ferroelectric TO1 (F1u); R25 =
# antiferrodistortive octahedral rotation; M3 = in-phase octahedral
# rotation; X5 = FE chain branch at X.
IRREPS = {
    ("SrTiO3", "Gamma"): "Gamma15 (F1u, FE)",
    ("SrTiO3", "X"): "X5 (FE branch)",
    ("SrTiO3", "M"): "M3 (AFD, in-phase rotation)",
    ("SrTiO3", "R"): "R25 (AFD rotation)",
    ("BaTiO3", "Gamma"): "Gamma15 (F1u, FE)",
    ("BaTiO3", "X"): "X5 (FE chain branch)",
    ("BaTiO3", "M"): "M3' (FE chain branch)",
    ("BaTiO3", "R"): "-",
}


def unstable_ranges(df: pd.DataFrame) -> list[str]:
    """Compress unstable path indices (any omega < -1 cm^-1) into ranges."""
    idxs = sorted(df[df.omega_cm1 < -1.0].path_index.unique())
    if not idxs:
        return []
    ranges, start, prev = [], idxs[0], idxs[0]
    for i in idxs[1:]:
        if i != prev + 1:
            ranges.append((start, prev))
            start = i
        prev = i
    ranges.append((start, prev))

    def name(i: int) -> str:
        if i in NODES:
            return NODES[i]
        seg = SEGMENTS[min(i // 40, 4)]
        return f"{seg}[{i % 40}/40]"

    return [f"{name(a)}..{name(b)}" for a, b in ranges]


def main() -> None:
    rows = []
    summaries = []
    for material in ("SrTiO3", "BaTiO3"):
        df = pd.read_csv(
            REPO / "data" / "processed" / f"harmonic_dispersion_{material}.csv", comment="#"
        )
        for index, label in NODES.items():
            if index == 120 or index == 200:  # repeated path nodes
                continue
            block = df[(df.path_index == index) & (df.omega_cm1 < -1.0)]
            if block.empty:
                continue
            q = block.iloc[0]
            omegas = block.omega_cm1.values
            rows.append(
                {
                    "material": material,
                    "q_label": label,
                    "qx": q.qx,
                    "qy": q.qy,
                    "qz": q.qz,
                    "degeneracy": len(omegas),
                    "abs_omega_cm1": round(float(-omegas.mean()), 1),
                    "irrep": IRREPS.get((material, label), "?"),
                }
            )
        summaries.append(f"#   {material}: unstable path regions: {'; '.join(unstable_ranges(df))}")

    out = REPO / "data" / "processed" / "instability_map.csv"
    header = [
        "# instability_map.csv - produced by scripts/instability_map.py",
        "# harmonic modes with omega^2 < 0 at high-symmetry points (|omega| in cm^-1;",
        "#   degeneracy = number of imaginary branches at that point; NAC direction (1,0,0)",
        "#   splits the Gamma triplet into doublet + hardened LO partner)",
        "# unstable regions along the full path (index/40 = fraction of segment):",
        *summaries,
    ]
    frame = pd.DataFrame(rows)
    out.write_text("\n".join(header) + "\n" + frame.to_csv(index=False))
    print(frame.to_string(index=False))
    for s in summaries:
        print(s.lstrip("# "))
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
