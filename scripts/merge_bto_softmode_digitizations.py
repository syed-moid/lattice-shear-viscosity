#!/usr/bin/env python3
"""Merge the cubic-BaTiO3 soft-mode digitizations into canonical files and
run the built-in consistency checks.

Raw inputs (NEVER edited; local-only, references/ is gitignored):
  references/digitized/vsr1982_fig3/raw/     — VSR PRB 26, 5904 (1982) Fig. 3
  references/digitized/vsr1982_fig6/raw/     — VSR Fig. 6
  references/digitized/presting1983_fig2/raw/— Presting PRB 28, 6097 (1983) Fig. 2
See the README.md beside each raw/ for the symbol->source maps and the
axis-unit verification (ALL temperature columns confirmed to be KELVIN
against the figure images — cubic-phase data begin at ~394-408 K, just
above T_C ~= 393 K; a Celsius axis would have begun near 120).

Canonical outputs (committed), long format (T_K, value, source):
  data/processed/bto_softmode_digitized/omega0_T.csv        [cm-1]
  data/processed/bto_softmode_digitized/rel_damping_T.csv   [gamma_full/Omega0]
  data/processed/bto_softmode_digitized/omega0sq_T.csv      [10^3 cm-2]
  data/processed/bto_softmode_digitized/invtau_T.csv        [cm-1]
  data/processed/bto_softmode_digitized/gamma_abs_T.csv     [cm-1, FULL damping]

Consistency checks (all outcomes logged, none suppressed):
  (a) calibration anchor: VSR this-work Omega0 near 473 K ~= 31 cm-1
      (VSR's own stated intensity-calibration value);
  (b) Presting Fig. 2 full symbols are a re-plot of VSR's own data:
      omega0_vsr1982(Presting) vs omega0_vsr1982_thiswork(Fig. 3), and
      gamma_vsr1982(Presting) vs (gamma/Omega0 x Omega0)(Fig. 3);
  (c) Fig. 6 top panel is labeled "1/tau = Omega0^2/gamma (cm-1)" — the
      convention factor is exactly 1, so invtau must track
      Omega0/(gamma/Omega0) from the Fig. 3 series;
  (d) omega0sq_fit (Fig. 6) must equal (omega0_thiswork)^2 (Fig. 3).
Plus the Harada pin: the neutron damping ratio gamma/omega0 ~= 2.3 at
423 K (q = 0.313 A^-1) vs the hyper-Raman zone-center ratio — a
cross-technique, finite-q-vs-q=0 comparison, reported as such.

Conventions are applied ONLY downstream (build_bto_softmode_inputs.py):
full gamma -> Gamma_HWHM = gamma/2; overdamped flags. This script merges
and checks; it converts no physics conventions.

Usage: uv run python scripts/merge_bto_softmode_digitizations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "references" / "digitized"
OUT = REPO / "data" / "processed" / "bto_softmode_digitized"


def load(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        x, y = (float(v) for v in line.split(","))
        rows.append((x, y))
    rows.sort()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def write_canonical(name: str, series: list[tuple[str, np.ndarray, np.ndarray]],
                    unit: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name}.csv - produced by scripts/merge_bto_softmode_digitizations.py",
             f"# units: {unit}. {note}",
             "# raw sources under references/digitized/ (local-only; see READMEs there",
             "# for symbol maps and the Kelvin axis verification). Raw files unedited.",
             "T_K,value,source"]
    for src, T, v in series:
        for t, val in zip(T, v):
            lines.append(f"{t:.2f},{val:.5g},{src}")
    (OUT / f"{name}.csv").write_text("\n".join(lines) + "\n")
    n = sum(len(T) for _, T, _ in series)
    print(f"  -> {name}.csv ({n} rows, {len(series)} sources)")


def main() -> None:
    f3 = RAW / "vsr1982_fig3" / "raw"
    f6 = RAW / "vsr1982_fig6" / "raw"
    p2 = RAW / "presting1983_fig2" / "raw"

    om_vsr = load(f3 / "omega0_vsr1982_thiswork.csv")
    om_lus = load(f3 / "omega0_luspin1980.csv")
    om_bar = load(f3 / "omega0_barker1966.csv")
    rd_vsr = load(f3 / "rel_damping_vsr1982_thiswork.csv")
    rd_lus = load(f3 / "rel_damping_luspin1980.csv")
    rd_bar = load(f3 / "rel_damping_barker1966.csv")
    o2_fit = load(f6 / "omega0sq_vsr1982_fit.csv")
    o2_int = load(f6 / "omega0sq_vsr1982_intensity.csv")
    itau = load(f6 / "invtau_vsr1982.csv")
    om_p_vsr = load(p2 / "omega0_vsr1982.csv")
    om_p_ino = load(p2 / "omega0_inoue_akimoto.csv")
    ga_p_vsr = load(p2 / "gamma_vsr1982.csv")
    ga_p_ino = load(p2 / "gamma_inoue_akimoto.csv")

    print("Canonical merged files:")
    write_canonical("omega0_T", [
        ("vsr1982_fig3_thiswork", *om_vsr),
        ("luspin1980_via_vsr_fig3", *om_lus),
        ("barker1966_via_vsr_fig3", *om_bar),
        ("vsr1982_via_presting_fig2", *om_p_vsr),
        ("inoue_akimoto1983_via_presting_fig2", *om_p_ino),
    ], "cm-1", "Soft-TO frequency Omega0, cubic phase.")
    write_canonical("rel_damping_T", [
        ("vsr1982_fig3_thiswork", *rd_vsr),
        ("luspin1980_via_vsr_fig3", *rd_lus),
        ("barker1966_via_vsr_fig3", *rd_bar),
    ], "dimensionless", "RELATIVE damping gamma_full/Omega0 (classical-oscillator FULL gamma).")
    write_canonical("omega0sq_T", [
        ("vsr1982_fig6_fit_PRIMARY", *o2_fit),
        ("vsr1982_fig6_intensity_corroboration", *o2_int),
    ], "10^3 cm-2", "Omega0^2; the intensity series assumes T-independent HR "
       "polarizability (calibrated Omega0=31 cm-1 at 473 K) — corroboration only.")
    write_canonical("invtau_T", [
        ("vsr1982_fig6", *itau),
    ], "cm-1", "Debye relaxation rate, panel-defined as 1/tau = Omega0^2/gamma.")
    write_canonical("gamma_abs_T", [
        ("vsr1982_via_presting_fig2", *ga_p_vsr),
        ("inoue_akimoto1983_via_presting_fig2", *ga_p_ino),
    ], "cm-1", "ABSOLUTE classical-oscillator FULL damping constant gamma.")

    # ------------- consistency checks -------------
    print("\nConsistency checks:")
    results = []

    # (a) calibration anchor
    om473 = float(np.interp(473.0, *om_vsr))
    ok_a = abs(om473 - 31.0) / 31.0 < 0.10
    results.append(("a", ok_a,
                    f"Omega0(473 K, VSR this-work) = {om473:.1f} cm-1 vs stated 31 "
                    f"({100 * (om473 / 31.0 - 1):+.1f}%)"))

    # (b) Presting re-plot vs Fig. 3
    Tb = om_p_vsr[0]
    om_f3_on_p = np.interp(Tb, *om_vsr)
    dev_om = (om_p_vsr[1] - om_f3_on_p) / om_f3_on_p
    ga_f3 = np.interp(Tb, *rd_vsr) * om_f3_on_p
    dev_ga = (ga_p_vsr[1] - np.interp(ga_p_vsr[0], Tb, ga_f3)) / np.interp(ga_p_vsr[0], Tb, ga_f3)
    ok_b = np.median(np.abs(dev_om)) < 0.05 and np.median(np.abs(dev_ga)) < 0.07
    results.append(("b", ok_b,
                    f"Presting-vs-Fig3: Omega0 median|dev| = {100 * np.median(np.abs(dev_om)):.1f}% "
                    f"(max {100 * np.max(np.abs(dev_om)):.1f}%); gamma median|dev| = "
                    f"{100 * np.median(np.abs(dev_ga)):.1f}% (max {100 * np.max(np.abs(dev_ga)):.1f}%)"))

    # (c) invtau vs Omega0/(gamma/Omega0), factor exactly 1 (panel label)
    Tc = itau[0]
    pred = np.interp(Tc, *om_vsr) / np.interp(Tc, *rd_vsr)
    dev_c = (itau[1] - pred) / pred
    ok_c = np.median(np.abs(dev_c)) < 0.10
    results.append(("c", ok_c,
                    f"1/tau vs Omega0^2/gamma (factor 1, panel label): median|dev| = "
                    f"{100 * np.median(np.abs(dev_c)):.1f}% (max {100 * np.max(np.abs(dev_c)):.1f}%)"))

    # (d) omega0sq_fit vs (omega0_thiswork)^2  [fit file is in 10^3 cm-2]
    Td = o2_fit[0]
    inside = (Td >= om_vsr[0].min()) & (Td <= om_vsr[0].max())
    pred_sq = np.interp(Td[inside], *om_vsr) ** 2 / 1e3
    dev_d = (o2_fit[1][inside] - pred_sq) / pred_sq
    ok_d = np.median(np.abs(dev_d)) < 0.12
    results.append(("d", ok_d,
                    f"Omega0^2(fit) vs Omega0(Fig3)^2: median|dev| = "
                    f"{100 * np.median(np.abs(dev_d)):.1f}% (max {100 * np.max(np.abs(dev_d)):.1f}%, "
                    f"{int(inside.sum())} common-T points)"))

    for tag, ok, msg in results:
        print(f"  ({tag}) [{'PASS' if ok else 'FAIL'}] {msg}")

    # Harada pin (cross-technique, finite-q vs q=0 — reported, not gated)
    rd423 = float(np.interp(423.0, *rd_vsr))
    print(f"\n  Harada pin: hyper-Raman zone-center gamma/Omega0(423 K) = {rd423:.1f} "
          f"vs neutron 2.24 at q = 0.313 A^-1 (softmode_inputs_BaTiO3.csv). The "
          f"zone-center response is ~{rd423 / 2.24:.1f}x more heavily damped than at "
          f"finite q — consistent with the strong q-dependence of the soft-mode "
          f"damping reported by Harada et al. themselves; a cross-technique, "
          f"different-q comparison, NOT a contradiction. Both are carried with "
          f"per-row sources downstream.")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{4 - n_fail}/4 checks PASS" + ("" if n_fail == 0 else " — INVESTIGATE FAILURES"))
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
