# Stage C — SrTiO₃ $\eta_{xyxy}$: provenance, audit (A), external consistency check (B) — 2026-07-24

(Framing language updated 2026-08-14 to match the manuscript's current epistemics: the
Maerten comparison is an order-of-magnitude consistency check of a longitudinal
measurement against the computed shear component, not an adjudication; the revised
expectation band is 1e-3..1e-2 Pa s. The historical narrative below is unchanged.)

Supersedes the 2026-07-23 first version of this report. Chronology of the
assembly's three generations, each change audit-driven and documented:

| generation | eta(300 K) | what changed |
|---|---|---|
| 1 | 1.89e-2 Pa s (non-monotonic in T) | first assembly; character-blind frequency-binned Gamma map |
| 2 | 5.75e-3 (monotonic) | character-aware Gamma for bare-unstable modes; Vogt 300 K edge tolerance |
| 3 (current) | **3.89e-3** (monotonic, smooth across the Vogt boundary) | A-audit fixes: per-q RANK PAIRING of the bare<->renormalized ALAMODE correspondence (the raw (q,branch) pairing mapped bare TO1 -> acoustic 0 at Gamma, poisoning the unstable region of the eigenvalue map); soft-character Gamma extended to the whole bare [5,50) cm⁻¹ manifold; physical branch-minimum floor omega_r >= omega_s(Gamma,T) (Vogt where covered, ALAMODE's own renormalized Gamma-point TO1 otherwise) |

## Formula and constants

$\eta_{xyxy}$(T) = (1/(V_cell N_q k_B T)) sum_qs (hbar w)^2 gamma_xy^2 n(n+1) tau_eff,
tau_eff = 1/(2[Gamma - Re sqrt(Gamma^2 - w^2)]) (overdamped-safe).
V_cell = (3.8930 A)^3 (PBEsol cell); N_q = 11^3; 15 branches; the 3
acoustic Gamma translations are excluded.

## A1 — unit-and-convention chain for every low-omega input

| quantity | source value/units | conversions applied | where |
|---|---|---|---|
| D = d(omega^2)/d(eps) | strained-cell .modes, cm⁻¹ (matdyn; imaginary printed NEGATIVE) | signed eigenvalue sign(w)*w^2 [cm-2] BEFORE differencing (no abs/sqrt anywhere; regression-guarded) | check_shear_nonlinearity.compute_dataset |
| Lambda (Route H) | = -D, same strained cells. NOT taken from Vogt — Vogt supplies only omega_s and damping | none | compute_eta_SrTiO3.assemble |
| gamma (Route S) | -D/(2*omega0^2), dimensionless (gamma = -d ln omega/d eps = -(1/2) d ln omega^2/d eps — the 1/2 is definitional, applied once) | none | assemble |
| gamma (Route H) | Lambda/(2*omega_r^2) = -D/(2*omega_r^2); reduces to Route S as omega_r -> omega0 | none | assemble |
| omega_r | ALAMODE SCPH-coupled .result frequencies, cm⁻¹, via the rank-paired eigenvalue map | cm⁻¹ -> rad/s by 2*pi*c*100 exactly once, at the weight/tau step | assemble |
| Gamma_anh | ALAMODE #GAMMA_EACH, cm⁻¹, HWHM (convention pinned earlier by unit-trace + Tadano cross-check) | cm⁻¹ -> rad/s once; tau = 1/(2*Gamma) via tau_effective — the single factor 2 of the HWHM convention, applied once | assemble |
| Gamma (Vogt, Gamma sector) | softmode_inputs_SrTiO3.csv `Gamma_HWHM_cm1` — Vogt's FULL gamma was halved ONCE at import (Vogt Eq. 13 convention, documented in that CSV header). The eta script consumes the HWHM column directly — no second halving, no omission | cm⁻¹ -> rad/s once | load_vogt/assemble |
| Gamma_iso | Tamura rate 1/tau_iso [rad/s] -> HWHM = rate/2 (once) -> cm⁻¹; Matthiessen Gamma_anh + Gamma_iso | as stated | compute_eta_isotope |
| weight | (hbar*w)^2 * n(n+1), w = omega_r in rad/s, n = Bose(hbar w / k_B T) | — | assemble |
| prefactor | 1/(V_cell * N_q * k_B * T) | dimensional check: J^2 * s / (J * m^3) = Pa s | assemble |

**Explicit caveat on the kinetic anchor (do not over-read the 0.96):**
the kinetic estimate eta_kin = 3 n_at k_B T <gamma^2 tau> is built from
the SAME per-mode gamma and tau as the full sum. A factor-2 (or any
global) slip in gamma, tau, or the linewidth convention would rescale
BOTH numerator and denominator of eta_full/eta_kin identically and leave
the ratio at ~1. The 0.96 therefore checks the mode-weighting and
prefactor plumbing of Eq. (5) — NOT the correctness of the input
conventions. The conventions are instead pinned by the table above and
by the external comparison in section B.

## A2 — double-count check (audit_eta_assembly.py)

19 965 (q,branch) slots = 3 acoustic-Gamma translations (skipped)
+ 13 053 Route S + 6 775 Route H stable + 132 Route H unstable
+ 2 Gamma/Vogt sector; every summed slot claimed exactly once —
**disjointness PASS, completeness PASS** (0 negative-mapped skips).

## A3 — sensitivity (current generation, 300 K)

Low-omega sector = 3.63e-3 of 3.89e-3. By omega0 bin (share of total
eta): [-inf,0): 3.4%; [5,25): 2.5%; [25,50): 12.3%; [50,75): 15.1%;
[75,100): 16.1%; [100,125): 35.0%; [125,150): 3.5%; [150,175): 5.6%.
Top-20 single modes carry 13.5% (largest single mode 0.8%) —
**broad-based**, with individually plausible inputs (gamma 10-20, tau
1.5-3.8 ps, Gamma 0.7-1.8 cm⁻¹). The generation-2 audit had exposed the
concentrated artifacts (4 near-Gamma modes at 15% with omega_r = 39.9
cm⁻¹ BELOW the measured branch minimum, and soft-branch modes with
acoustic tau = 4.7 ps); the generation-3 fixes removed both — root cause
of the omega_r artifact was the Gamma-point rank-pairing corruption, not
interpolation.

## A4 — regression tests

`tests/test_eta_assembly.py`: (1) Gamma-point rank pairing must survive
the bare/renormalized branch-ordering swap (bare TO1 pairs with
renormalized TO1, floor = renormalized TO1); (2) character-aware Gamma
keeps the soft and acoustic populations separate where they overlap in
renormalized frequency (a frequency-blind median is demonstrably wrong
for both). 13/13 suite passes.

## B — external consistency check (gigahertz acoustic damping)

Sources scanned for STO acoustic attenuation / mechanical Q near 300 K:
every PDF in both literature folders (Fauque 2022 INS — TA dispersion
softening, ultrasound mentioned only as 20-140 K velocity comparisons;
Akimov 2000 — film Raman, no acoustics; Schmidt 2025 — SSCHA theory;
Bussmann-Holder 2024 review, Maity 2025, Verdi 2023, Vogt 1995,
Tadano 2015, Yamada-Shirane 1969 — keyword scans negative). **None
contains a direct attenuation/Q number.** The decisive source was
located open-access instead:

**Maerten, Bojahr, Reinhardt, Koreeda, Roessle, Bargheer,
"Critical behavior of the damping rate of GHz acoustic phonons in
SrTiO3...", arXiv:1810.00381 (2018)** — time- and frequency-resolved
Brillouin scattering of LA phonons in bulk-like STO substrates.
Key measured facts (quoted from the paper):
- LA phonons at q = 52-58 um^-1 (f ~ 70-74 GHz), v_L ~ 7.9-8.1 nm/ps.
- "The phonon damping is in our samples at 300 K on the order of
  1-2 GHz"; bulk STO / BS values "~1 GHz"; LSMO-transducer TDBS
  "T independent value of Gamma ~ 2 GHz" in the cubic phase.
- **Fig. 6: the damping follows Akhiezer's q^2 law at 300 K across
  q = 0.4-100 um^-1**, connecting their GHz data to the older
  low-frequency ultrasonic points (Nava et al., Nagakubo et al.) — the
  frequency-scaling assumption (alpha ∝ omega^2) is experimentally
  verified at this temperature over ~2 decades in q, which is exactly
  the Akhiezer-regime validity statement our conversion needs.
- Their Gamma is the amplitude decay rate of the TDBS oscillation
  (= angular HWHM of the BS line: beta = Gamma/2*pi, their Fig. 3),
  so the viscous-damping relation is Gamma_amp = alpha*v =
  omega^2 * eta / (2 rho v^2).

Implied viscosity from their 300 K measurements (rho = 5110 kg/m^3,
v = 8000 m/s, q = 58 um^-1 -> omega = v*q = 4.64e11 rad/s):

| measurement | Gamma_amp (GHz = 1e9 s^-1) | implied eta (Pa s) |
|---|---|---|
| bulk STO substrate, BS | ~1 | **3.0e-3** |
| LSMO sample, TDBS, cubic phase | ~2 | **6.1e-3** |
| pre-registered decade upper edge (1e-3) | would require 0.33 | (not observed) |
| pre-registered decade lower edge (1e-4) | would require 0.033 | (not observed) |
| **our $\eta_{xyxy}$(300 K)** | (predicts 1.28 at their q) | **3.89e-3** |

**Outcome: experiment sits at (3-6)e-3 Pa s — i.e., nearer 1e-2 than
1e-3 — consistent with, and bracketing, our 3.89e-3** (factor 0.77-1.6
of the measured range). Equivalently: our predicted damping at their wavevector,
1.28 GHz, lies inside their measured 1-2 GHz. The formerly expected
1e-4..1e-3 decade corresponds to damping rates 3-30x SMALLER than
anything measured — below every measured point (the expectation band
was accordingly revised to 1e-3..1e-2 Pa s). Two caveats, stated so
they are not lost: (i) their phonons are LONGITUDINAL along [100], so
the measured combination is the longitudinal viscosity eta_xxxx, while
ours is the shear component $\eta_{xyxy}$ — the comparison is
order-of-magnitude-exact only (computing eta_xxxx needs the tetragonal
strain derivative gamma_xx, i.e. a uniaxial-strain pair we have not
run); (ii) sample-to-sample spread (bare substrate 1 GHz vs
transducer-covered 2 GHz) bounds the experimental systematic at ~2x.
Also logged, not cherry-picked: no measurement disagrees with our value
at worse than the factor-1.6 above; the older ultrasonic points
(Nava, Nagakubo) lie ON the q^2 line through the GHz data in the
paper's own Fig. 6, so they imply the same eta within its scatter.

At 1 GHz our (corrected) eta gives alpha = 11.1 dB/cm and
Q^-1 = 2.0e-4.

## Corrected results (generation 3)

| T (K) | eta_total (Pa s) | Route S | H stable | H unstable | Gamma sector | flags |
|---|---|---|---|---|---|---|
| 100 | 6.03e-3 | 9.7e-5 | 4.24e-3 | 1.58e-3 | 1.2e-4 | provisional-method |
| 150 | 4.68e-3 | 1.7e-4 | 3.87e-3 | 6.1e-4 | 2.9e-5 | provisional-method |
| 200 | 4.19e-3 | 2.2e-4 | 3.67e-3 | 2.9e-4 | 1.2e-5 | provisional-method |
| 250 | 3.95e-3 | 2.3e-4 | 3.54e-3 | 1.8e-4 | 7.3e-6 | provisional-method |
| 300 | 3.89e-3 | 2.6e-4 | 3.50e-3 | 1.3e-4 | 5.2e-6 | — |
| 350 | 3.68e-3 | 2.8e-4 | 3.39e-3 | 1.0e-5 | 2.0e-6 | provisional-method + Vogt-fallback (theory floor used) |
| 400 | 3.62e-3 | 2.8e-4 | 3.33e-3 | 8.5e-6 | 1.6e-6 | provisional-method + Vogt-fallback (theory floor used) |

Isotope series (regenerated at generation 3): eta monotonically
decreasing, -7.0% at f = 0.15 (3.886 -> 3.614e-3); kinetic anchor ratio
0.96 (see the A1 caveat on what that does and does not check).

## Remaining approximations (unchanged status)

- Frequency-map layer instead of true cross-code branch matching
  (rank-paired per q, character-split; upgrade path = eigenvector-level
  matching between the QE and ALAMODE cells).
- Route S gamma at bare omega0 (small in that manifold); weights at
  mapped omega_r.
- Isotope site projection by total DOS (upper bound on suppression).
- T != 300 K linewidths method-validated only; Vogt series ends at
  298 K (theory floor takes over above).
- eta_xxxx (longitudinal) not computable from current data — would need
  a uniaxial strain pair; relevant to sharpen the B-comparison.
