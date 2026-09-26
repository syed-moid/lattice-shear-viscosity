# Stage C — BaTiO₃ zone-center (Route H) viscosity assembly: provenance and scale-test report (2026-07-24; tau updated 2026-08-14; conventions revised 2026-09-22)

**Revision update (2026-09-22, numbers only).** (D1) Λ is now the tensor strain derivative
dω²/dε_xy = ∓0.58×10⁵ cm⁻² (the July ∓1.17×10⁵ was the derivative along the symmetric path ε_xy = ε_yx = s,
2× the tensor value), so every sector value is 1/4 of the July one; (D2) the lifetime is the stress-correlator
two-pole kernel τ = 1/(2Γ) + 2Γ/ω² (exact classical time integral of the stiffness-conjugate stress correlator
of an effective damped oscillator with friction 2Γ; every zone-center point is deeply overdamped, Γ/ω_s =
1.3–4.2, where this kernel exceeds the energy-variable form by 2.6–3.5×), so the series is 2.6–3.5× the
August one and the 700→410 K rise is 171-fold; (D5) the result is an exploratory sector quantity — the
"scale-expectation test" and the STO-share-scaled inference below are withdrawn, no total-viscosity value is
inferred; (D9) T_C of the VSR (melt-grown) samples is 403 K, so 410 K is T_C + 7 K. The damping column
Gamma_HWHM = γ_VSR/2 is the DHO friction parameter Γ of ẍ + 2Γẋ + …, not a spectral half width. Regenerated
table: `data/processed/eta_BaTiO3.csv`.

| T (K) | omega_s (cm⁻¹) | Gamma (cm⁻¹, DHO friction) | eta44_soft_sector (Pa s), 2026-09-22 | cap x1.5 |
|---|---|---|---|---|
| 410 | 10.8 | 45.4 | 5.02e-4 | +0% |
| 425 | 16.1 | 45.7 | 1.66e-4 | +1% |
| 450 | 25.2 | 52.0 | 5.41e-5 | +3% |
| 475 | 32.6 | 56.0 | 2.95e-5 | +5% |
| 500 | 43.5 | 77.2 | 1.67e-5 | +8% |
| 550 | 54.6 | 86.7 | 1.01e-5 | +12% |
| 600 | 67.1 | 97.1 | 6.19e-6 | +19% |
| 650 | 82.6 | 118.8 | 3.79e-6 | +28% |
| 700 | 91.2 | 122.3 | 2.94e-6 | +34% |

The historical narrative below (August numbers, path convention, energy-variable kernel) is unchanged.

(2026-08-14 update: the lifetime entering the sector integral was changed from
the slow-pole form to the exact two-pole closed form
tau = (Gamma^2 + omega^2)/(2 Gamma omega^2) — latvisc.viscosity.tau_two_pole_exact,
manuscript Appendix A.3 — and the series below was regenerated. The exact form
reduces the most-overdamped near-T_C points (which the slow-pole form
overweighted) and slightly raises the underdamped-dominated high-T points.)

Scope: **Gamma-point-only** (manuscript §3.2) — after the fc3 cost-gate
NO-GO, BaTiO₃ has no full-zone linewidth map; the computable quantity
with stated provenance is the soft-TO-branch sector of $\eta_{44}$ anchored
entirely at the zone center. This is a PARTIAL viscosity (lower bound
with respect to the full-zone sum).

## Input provenance (every number traced)

| ingredient | value/source | notes |
|---|---|---|
| Lambda_b (shear coupling, soft TO doublet at Gamma) | -117 768 / +116 408 cm-2 per unit strain — OWN PBEsol strained cells (D of the bare-imaginary doublet, BTO shear pair) | symmetric E-type split; sum = -1360 (1.2% of magnitude) ~ 0 as symmetry requires — internal check PASS |
| omega_s(T) | VSR 1982 hyper-Raman Fig. 3, "this work" series (PRIMARY), 12 measured points 408-706 K, interpolated linearly | NO Cochran/Curie-Weiss fit imposed (VSR observe systematic deviation from the linear law); calibration anchor 31.7 cm⁻¹ at 473 K vs stated 31 (+2.3%) — check (a) PASS |
| Gamma_s(T) HWHM | VSR Fig. 3 gamma/Omega0 x Omega0 / 2 | full->HWHM halving applied ONCE, in code; all points overdamped (gamma_full/Omega0 = 2.7-8.5) |
| soft-branch dispersion | Harada 1971 neutron: A_par = 972 meV^2A^2 ([100], +/-20%), A_perp = 4750 (stiff [110]/[111]) | self-check: predicts omega(0.313 A^-1, 423 K) = 79 cm⁻¹ = Harada's own measured point |
| integration cap | q_par <= 0.47 A^-1 (Harada's measured-dispersion range) | cap sensitivity reported per T (below) |
| tau | exact two-pole (Gamma^2+omega^2)/(2 Gamma omega^2) (`latvisc.viscosity.tau_two_pole_exact`) | all zone-center points deep in the overdamped regime; exact form valid uniformly, no regime switch |
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

| T (K) | omega_s (cm⁻¹) | Gamma_HWHM (cm⁻¹) | eta_soft (Pa s) | cap x1.5 |
|---|---|---|---|---|
| 410 | 10.8 | 45.4 | 5.74e-4 | +1% |
| 425 | 16.1 | 45.7 | 2.12e-4 | +2% |
| 450 | 25.2 | 52.0 | 7.83e-5 | +5% |
| 475 | 32.6 | 56.0 | 4.61e-5 | +9% |
| 500 | 43.5 | 77.2 | 2.48e-5 | +13% |
| 550 | 54.6 | 86.7 | 1.55e-5 | +19% |
| 600 | 67.1 | 97.1 | 9.71e-6 | +27% |
| 650 | 82.6 | 118.8 | 5.78e-6 | +37% |
| 700 | 91.2 | 122.3 | 4.59e-6 | +43% |

The **x125 rise from 700 K to 410 K is the critical enhancement of
section 2.6 realized with measured inputs**: gamma_soft =
Lambda/(2 omega_s^2) grows from ~7 to ~500 as omega_s collapses from 91
to 11 cm⁻¹, tamed but not canceled by the critical slowing of the exact
two-pole tau. Near T_C the cap sensitivity vanishes (the integrand is
concentrated at small q); away from T_C it grows to +43% — there the
SECTOR is small anyway and the uncapped tail belongs physically to the
stable manifold this scope excludes.

## SCALE-EXPECTATION TEST — outcome, unadjusted

Revised expectation under test (2026-07-24 sign-off, STO-grounded; see
eta_SrTiO3_stageC.md section B): full-zone eta in 1e-3..1e-2 Pa s.
Outcome at Gamma-point-only scope (exact two-pole tau, 2026-08-14):

- The sector value alone reaches 5.7e-4 Pa s at 410 K — within a factor
  of two of the band's lower edge, from ONE branch family — and falls
  to 4.6e-6 by 700 K.
- The sector CANNOT test the full-zone expectation on its own: in
  SrTiO₃ the equivalent (formerly-unstable + Gamma) sector carries only
  ~3.5% of the full sum at 300 K. Scaling by that share gives a
  full-zone INFERENCE of ~1.6e-2 Pa s at 410 K — ABOVE the band — but
  the share is strongly T-dependent (26% at 100 K in SrTiO₃, deep in
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
