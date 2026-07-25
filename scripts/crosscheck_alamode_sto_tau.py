#!/usr/bin/env python3
"""Cross-check a local ALAMODE SrTiO3 RTA run against Tadano & Tsuneyuki
(2015), PRB 92, 054301, Fig. 7.

Parses a .result file (frequencies + per-mode Gamma_q(T), HWHM in cm-1 per
this project's pinned convention), computes tau_q(300K) = 1/(2*Gamma_q),
and reports a band-by-band comparison against the digitized reference
table in data/processed/reports/tadano2015_tau300K_crosscheck.md.

Reads : data/raw/alamode_sto/<name>.result (default: STO_RTA_production.result,
        the bare-RTA run; pass a different filename for the SCPH-coupled run)
Writes: data/processed/reports/alamode_sto_tau300K_bands<_suffix>.csv

Usage: uv run python scripts/crosscheck_alamode_sto_tau.py [result_filename] [output_suffix]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.constants import speed_of_light

REPO = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO / "data" / "raw" / "alamode_sto"
OUT_DIR = REPO / "data" / "processed" / "reports"
CM1_TO_RAD_PER_S = 2.0 * np.pi * speed_of_light * 100.0
PS = 1e12

# Digitized Tadano & Tsuneyuki (2015) Fig. 7 tau(300K) inset bands, see
# data/processed/reports/tadano2015_tau300K_crosscheck.md
BANDS = [
    (0, 50, "5-20 ps (long-lived acoustic tail, max ~15-20)"),
    (50, 100, "(gap between digitized bands -- not in the paper's own table;"
              " added after finding the longest-tau SCPH-coupled mode, 17.5 ps"
              " at 54.9 cm-1, fell here)"),
    (100, 150, "1-4 ps (dense cluster)"),
    (150, 200, "0.7-3 ps (declining)"),
    (200, 450, "0.4-1.2 ps (broad plateau ~0.6-0.8)"),
    (460, 490, "1-2 ps (local rise)"),
    (500, 550, "0.12-0.5 ps (pronounced dip, min ~0.12)"),
    (560, 730, "spectral gap, no modes expected"),
    (730, 830, "0.2-0.45 ps (top optic branch)"),
]


def parse_result(path: Path, target_temp: int = 300):
    """Parse frequencies and Gamma_q(target_temp). Detects the temperature
    grid from the #TEMPERATURE block so this works for both the 21-point
    bare-RTA scan and a single-temperature SCPH-coupled run."""
    text = path.read_text()
    freq = {}
    freq_section = re.search(
        r"##Phonon Frequency\n#[^\n]*\n(.*?)##END Phonon Frequency", text, re.S
    ).group(1)
    for line in freq_section.strip().splitlines():
        q, b, w = line.split()
        freq[(int(q), int(b))] = float(w)

    tmin, tmax, tstep = (float(x) for x in
                         re.search(r"#TEMPERATURE\n([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)", text).groups())
    temps = [tmin] if tstep == 0 or tmax == tmin else list(np.arange(tmin, tmax + tstep / 2, tstep))
    t_idx = min(range(len(temps)), key=lambda i: abs(temps[i] - target_temp))
    assert abs(temps[t_idx] - target_temp) < 1e-6, f"{target_temp}K not in temperature grid {temps}"

    gamma_t = {}
    for q, b, mult, body in re.findall(
        r"#GAMMA_EACH\n(\d+) (\d+)\n(\d+)\n((?:.*\n)*?)#END GAMMA_EACH", text
    ):
        q, b, mult = int(q), int(b), int(mult)
        lines = body.strip().splitlines()
        vals = [float(x) for x in lines[mult:]]  # skip `mult` velocity-triplet lines
        assert len(vals) == len(temps), f"{q},{b}: expected {len(temps)} T values, got {len(vals)}"
        gamma_t[(q, b)] = vals[t_idx]
    return freq, gamma_t


def main() -> None:
    result_name = sys.argv[1] if len(sys.argv) > 1 else "STO_RTA_production.result"
    suffix = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    result_path = RESULT_DIR / result_name
    out_path = OUT_DIR / f"alamode_sto_tau300K_bands{suffix}.csv"

    freq, gamma300 = parse_result(result_path)
    print(f"Source: {result_path.relative_to(REPO)}")
    print(f"Parsed {len(freq)} (q,branch) frequencies, {len(gamma300)} GAMMA_EACH blocks")

    omegas = np.array(list(freq.values()))
    n_neg = int(np.sum(omegas < 0))
    q_with_unstable = {q for (q, _), w in freq.items() if w < 0}
    n_q = len({q for q, _ in freq})
    print(f"\nStability audit: {n_neg}/{len(omegas)} (q,branch) entries have omega < 0 "
          f"(imaginary/unstable bare harmonic modes) = {100 * n_neg / len(omegas):.1f}%")
    print(f"Irreducible q-points with >=1 unstable branch: {len(q_with_unstable)}/{n_q}")

    rows = []
    for key, w in freq.items():
        if w <= 0:
            continue
        g = gamma300.get(key)
        if g is None or g <= 0:
            continue
        tau_ps = 1.0 / (2.0 * g * CM1_TO_RAD_PER_S) * PS
        rows.append((w, tau_ps))
    rows.sort()
    omega_arr = np.array([r[0] for r in rows])
    tau_arr = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} stable modes with tau(300K) computed "
          f"(excluded {len(omegas) - len(rows)} unstable/zero-gamma entries)")

    print("\nband-by-band comparison vs Tadano & Tsuneyuki (2015) Fig 7 inset:")
    print(f"{'band (cm-1)':>14} {'n_modes':>8} {'tau median (ps)':>16} {'tau range (ps)':>20}  paper")
    csv_rows = ["band_lo_cm1,band_hi_cm1,n_modes,tau_median_ps,tau_p10_ps,tau_p90_ps,paper_range_ps"]
    for lo, hi, label in BANDS:
        mask = (omega_arr >= lo) & (omega_arr < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"{f'[{lo},{hi})':>14} {n:8d} {'--':>16} {'--':>20}  {label}")
            csv_rows.append(f"{lo},{hi},0,,,,\"{label}\"")
            continue
        med = float(np.median(tau_arr[mask]))
        lo_t, hi_t = (float(x) for x in np.percentile(tau_arr[mask], [10, 90]))
        print(f"{f'[{lo},{hi})':>14} {n:8d} {med:16.3f} {f'[{lo_t:.3f},{hi_t:.3f}]':>20}  {label}")
        csv_rows.append(f"{lo},{hi},{n},{med:.4f},{lo_t:.4f},{hi_t:.4f},\"{label}\"")

    print(f"\nglobal tau envelope: [{tau_arr.min():.3f}, {tau_arr.max():.3f}] ps "
          f"(paper: [0.12, 20] ps)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# {out_path.name} - produced by scripts/crosscheck_alamode_sto_tau.py",
        f"# tau_q(300K) = 1/(2*Gamma_q), Gamma HWHM cm-1, from {result_path.relative_to(REPO)}",
        "# compare against data/processed/reports/tadano2015_tau300K_crosscheck.md's digitized Fig 7 inset",
    ]
    out_path.write_text("\n".join(header) + "\n" + "\n".join(csv_rows) + "\n")
    print(f"-> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
