# Stage C — BaTiO3 zone-center (Route H) viscosity assembly: provenance and gate report (2026-07-24)

Scope: **Gamma-point-only** (manuscript §3.2) — after the fc3 cost-gate
NO-GO, BaTiO3 has no full-zone linewidth map; the computable quantity
with stated provenance is the soft-TO-branch sector of eta_44 anchored
entirely at the zone center. This is a PARTIAL viscosity (lower bound
with respect to the full-zone sum).

## Input provenance (every number traced)

| ingredient | value/source | notes |
|---|---|---|
| Lambda_b (shear coupling, soft TO doublet at Gamma) | -117 768 / +116 408 cm-2 per unit strain — OWN PBEsol strained cells (D of the bare-imaginary doublet, BTO shear pair) | symmetric E-type split; sum = -1360 (1.2% of magnitude) ~ 0 as symmetry requires — internal check PASS |
| omega_s(T) | VSR 1982 hyper-Raman Fig. 3, "this work" series (PRIMARY), 12 measured points 408-706 K, interpolated linearly | NO Cochran/Curie-Weiss fit imposed (VSR observe systematic deviation from the linear law); calibration anchor 31.7 cm-1 at 473 K vs stated 31 (+2.3%) — check (a) PASS |
| Gamma_s(T) HWHM | VSR Fig. 3 gamma/Omega0 x Omega0 / 2 | full->HWHM halving applied ONCE, in code; all points overdamped (gamma_full/Omega0 = 2.7-8.5) |
| soft-branch dispersion | Harada 1971 neutron: A_par = 972 meV^2A^2 ([100], +/-20%), A_perp = 4750 (stiff [110]/[111]) | self-check: predicts omega(0.313 A^-1, 423 K) = 79 cm-1 = Harada's own measured point |
| integration cap | q_par <= 0.47 A^-1 (Harada's measured-dispersion range) | cap sensitivity reported per T (below) |
| tau_eff | overdamped-safe Eq. (12) | all zone-center points are deep in the overdamped branch |
| conversions | measured rho = 5990 kg/m^3, v_TA[100] = 4246 m/s (Li et al. 1991, cubic 443 K) | per the section-3.4 measured-velocities convention |

Digitization QA: 13 raw WebPlotDigitizer files organized under
`references/digitized/{vsr1982_fig3,vsr1982_fig6,presting1983_fig2}/raw/`
with per-figure READMEs (symbol->source maps, axis-unit verification:
ALL temperature axes confirmed Kelvin against the figure images).
Canonical merged files: `data/processed/bto_softmode_digitized/*.csv`
(T_K, value, source; raw files never edited). **All four built-in
consistency checks PASS** (`merge_bto_softmode_digitizations.py`):
(a) calibration anchor +2.3%; (b) Presting-Fig.2-vs-VSR-Fig.3 re-plot,
median deviations 1.3% (omega) / 1.0% (gamma); (c) 1/tau vs
Omega0^2/gamma with the panel-label factor of exactly 1, median 1.5%;
(d) Omega0^2(fit) vs Omega0^2(Fig. 3), median 1.5%. Harada pin:
zone-center gamma/Omega0(423 K) = 6.1 vs neutron 2.24 at q = 0.313
A^-1 — a finite-q-vs-q=0, cross-technique comparison consistent with
the strong q-dependence of the damping, not a contradiction.

## Results (`data/processed/eta_BaTiO3.csv`)

| T (K) | omega_s (cm-1) | Gamma_HWHM (cm-1) | eta_soft (Pa s) | cap x1.5 |
|---|---|---|---|---|
| 410 | 10.8 | 45.4 | 9.15e-4 | +0% |
| 425 | 16.1 | 45.7 | 2.76e-4 | +1% |
| 450 | 25.2 | 52.0 | 8.00e-5 | +5% |
| 475 | 32.6 | 56.0 | 4.02e-5 | +9% |
| 500 | 43.5 | 77.2 | 2.26e-5 | +11% |
| 550 | 54.6 | 86.7 | 1.27e-5 | +17% |
| 600 | 67.1 | 97.1 | 7.34e-6 | +26% |
| 650 | 82.6 | 118.8 | 4.47e-6 | +31% |
| 700 | 91.2 | 122.3 | 3.28e-6 | +40% |

The **x280 rise from 700 K to 410 K is the critical enhancement of
section 2.6 realized with measured inputs**: gamma_soft =
Lambda/(2 omega_s^2) grows from ~7 to ~500 as omega_s collapses from 91
to 11 cm-1, tamed but not canceled by the overdamped tau_eff. Near T_C
the cap sensitivity vanishes (the integrand is concentrated at small q);
away from T_C it grows to +40% — there the SECTOR is small anyway and
the uncapped tail belongs physically to the stable manifold this scope
excludes.

## PRE-REGISTERED TEST — outcome, unadjusted

Revised expectation under test (GATE_1p 2026-07-24, STO-grounded):
full-zone eta in 1e-3..1e-2 Pa s. Outcome at Gamma-point-only scope:

- The sector value alone reaches 9.2e-4 Pa s at 410 K — at the band's
  lower edge from ONE branch family — and falls to 3e-6 by 700 K.
- The sector CANNOT test the full-zone expectation on its own: in
  SrTiO3 the equivalent (formerly-unstable + Gamma) sector carries only
  ~3.5% of the full sum at 300 K. Scaling by that share gives a
  full-zone INFERENCE of ~2.6e-2 Pa s at 410 K — ABOVE the band — but
  the share is strongly T-dependent (26% at 100 K in SrTiO3, deep in
  its own near-transition regime) and 410 K is T_C + 17 K, where the
  soft sector should dominate, making the flat-share scaling an
  overestimate of unknown size.
- **Honest verdict: the test is INCONCLUSIVE at this scope near T_C,
  and UNTESTABLE away from T_C (sector too small to constrain the full
  sum). No pass/fail is claimed; nothing was tuned. What the sector
  result DOES establish: the near-T_C critical enhancement is real,
  large, and of the magnitude class that critical ultrasonic-attenuation
  anomalies at ferroelectric transitions suggest (qualitative; no
  digitized BTO attenuation data in hand). The decision on how to
  frame the test outcome in section 4.5 rests with the author.**

## Known limitations (restated)

Single-oscillator parameterization of a two-component response
(Presting Fig. 3 qualitative; Hlinka 2008); Gamma(q) = Gamma_s
(zone-center) everywhere; stable manifold excluded by scope; Lambda
from bare strained cells applied to renormalized denominators (the
Route-H prescription); A_par carries Harada's +/-20%.
