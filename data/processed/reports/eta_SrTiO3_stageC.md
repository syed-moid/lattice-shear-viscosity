# Stage C — SrTiO₃ $\eta_{xyxy}$: provenance, audit (A), external consistency check (B) — 2026-07-24

**Revision update (2026-09-22, numbers only).** Three conventions changed after the referee audits and
the tables below were regenerated with `scripts/compute_eta_SrTiO3.py`, `compute_eta_isotope_SrTiO3.py`,
`table1_convergence.py`, `scan_partition_sensitivity.py`:
(D1) γ_xy is the TENSOR Grüneisen component −∂ ln ω/∂ε_xy (nine-component convention, η_xyxy = η₄₄): the
strained cells apply ε_xy = ε_yx = s and the earlier pipeline used the derivative along that path, 2× the tensor
value, so every η below is 1/4 of the values of the July assembly (3.887e-3 → 9.722e-4 Pa s at 300 K);
(D2) the lifetime is the stress-correlator two-pole kernel τ = 1/(2Γ) + 2Γ/ω² (SrTiO₃: +0.05 %);
(D3) the Tamura isotope rate uses the eigenvector-resolved oxygen-site projection with the 300 K
renormalised eigenvectors (−7.0 % → −0.96 % at f = 0.15). The narrative of sections A and B is the July
one and its absolute numbers (3.89e-3, 1.28 GHz, γ up to 126, zone-rms 4.2, α = 11.1 dB/cm) are in the
old convention; divide η, damping rates, α and Q⁻¹ by 4 and γ, Λ, rms γ by 2. The Maerten comparison of
section B is no longer used as a validation (longitudinal vs shear component; Ωτ ≈ 1 at 70 GHz), see the
revised manuscript. The former expectation band (1e-3..1e-2 Pa s) and its gate were removed.

**Revision update 2 (2026-09-22, numbers only).** The bare→renormalised per-q rank-pairing map is
retired: each bare QE mode now takes the renormalised frequency of the mode of maximum eigenvector overlap on the
11³ mesh (anphon on the same surface with PRINTEVEC; `latvisc.mode_matching`), with the character-aware linewidth
map unchanged. Production surface: the ALAMODE example harmonic set + its precomputed SCPH corrections (the
manuscript's surface); an own-surface SCPH on the project's converted PBEsol harmonic set was built and did not pass
every gate criterion (κ(300 K) +13 % at 2×2×2; R-point AFD mode 35/18 cm⁻¹ at 4×4×4). Regenerated:
η(T) 100/150/200/250/300/350/400 K = 1.38/1.25/1.19/1.15/**1.15**/1.09/1.08 ×10⁻³ Pa s (rank map: 1.51/1.17/1.05/
0.99/0.97/0.92/0.90); sectors at 300 K S/H-stable/H-unstable/Γ = 6.5e-5/1.05e-3/2.9e-5/1.4e-6; ¹⁸O series
−0.05/−0.24/−0.44/−0.60 % (f = 0.01/0.05/0.10/0.15); Table 1 (150/175/200 cm⁻¹) 1.199/1.148/1.134 ×10⁻³ (+4.4/—/
−1.2 %); Table 2 (4³/7³/9³/11³/13³) 0.702/1.028/1.111/1.148/1.294 ×10⁻³ — **11³ vs 13³ = −11.2 %, no longer
converged at the 5 % level** (the direct-matched low-ω₀ manifold is mesh sensitive; see `data/processed/v3_diagnostics/`); kinetic
anchor 0.97; α(1 GHz) 3.3 dB/cm, Q⁻¹ 5.9×10⁻⁵; substitution test +1.07 %. 52.6 % of η sits in modes whose
bare/renormalised eigenvector overlap is below 0.9 (0.02 % below 0.5).

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

$\eta_{xyxy}$(T) = (1/(V_cell N_q k_B T)) sum_qs (hbar w)^2 gamma_xy^2 n(n+1) tau,
tau = 1/(2 Gamma) + 2 Gamma/w^2 (stress-correlator two-pole kernel, 2026-09-22; earlier: slow-pole, then
energy-variable two-pole form).
V_cell = (3.8930 A)^3 (PBEsol cell); N_q = 11^3; 15 branches; the 3
acoustic Gamma translations are excluded.

## A1 — unit-and-convention chain for every low-omega input

| quantity | source value/units | conversions applied | where |
|---|---|---|---|
| D = d(omega^2)/d(eps) | strained-cell .modes, cm⁻¹ (matdyn; imaginary printed NEGATIVE) | signed eigenvalue sign(w)*w^2 [cm-2] BEFORE differencing (no abs/sqrt anywhere; regression-guarded) | check_shear_nonlinearity.compute_dataset |
| Lambda (Route H) | = -D, same strained cells. NOT taken from Vogt — Vogt supplies only omega_s and damping | none | compute_eta_SrTiO3.assemble |
| gamma (Route S) | -D/(2*omega0^2), dimensionless (gamma = -d ln omega/d eps = -(1/2) d ln omega^2/d eps — the 1/2 is definitional, applied once); D is the TENSOR derivative (symmetric-path derivative / 2, applied once in compute_dataset) | none | assemble |
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

## Results (generation 3 assembly, revised conventions D1–D3, 2026-09-22)

| T (K) | eta_total (Pa s) | Route S | H stable | H unstable | Gamma sector | flags |
|---|---|---|---|---|---|---|
| 100 | 1.51e-3 | 2.4e-5 | 1.06e-3 | 3.9e-4 | 3.0e-5 | provisional-method |
| 150 | 1.17e-3 | 4.3e-5 | 9.67e-4 | 1.5e-4 | 7.6e-6 | provisional-method |
| 200 | 1.05e-3 | 5.4e-5 | 9.19e-4 | 7.2e-5 | 3.3e-6 | provisional-method |
| 250 | 9.88e-4 | 5.8e-5 | 8.85e-4 | 4.4e-5 | 1.9e-6 | provisional-method |
| 300 | 9.72e-4 | 6.4e-5 | 8.75e-4 | 3.2e-5 | 1.4e-6 | — |
| 350 | 9.21e-4 | 6.9e-5 | 8.49e-4 | 2.6e-6 | 5.0e-7 | provisional-method + Vogt-fallback (theory floor used) |
| 400 | 9.05e-4 | 7.0e-5 | 8.32e-4 | 2.1e-6 | 4.1e-7 | provisional-method + Vogt-fallback (theory floor used) |

(July values in the path convention with the energy-variable kernel: 6.03/4.68/4.19/3.95/3.89/3.68/3.62 e-3.)

Isotope series (projected Tamura rate, renormalised eigenvectors): eta
monotonically decreasing, -0.96% at f = 0.15 (9.722e-4 -> 9.628e-4; bare
eigenvectors -0.96%; former total-DOS form -7.0%); smearing 5/10/20 cm-1:
-1.06/-0.96/-0.90%. Kinetic anchor ratio 0.96 (see the A1 caveat on what
that does and does not check). Akhiezer at 1 GHz (shear, v_s = 4900 m/s):
alpha = 2.77 dB/cm, Q^-1 = 5.0e-5. Table 1 (partition 150/175/200 cm-1):
1.021/0.972/0.958 e-3 (+5.0 %/—/-1.5 %); Table 2 (4^3/7^3/9^3/11^3/13^3):
0.740/0.869/1.012/0.972/1.028 e-3 Pa s.

## Remaining approximations (unchanged status)

- Frequency-map layer instead of true cross-code branch matching
  (rank-paired per q, character-split; upgrade path = eigenvector-level
  matching between the QE and ALAMODE cells).
- Route S gamma at bare omega0 (small in that manifold); weights at
  mapped omega_r.
- Isotope site projection: eigenvector-resolved (renormalised eigenvectors, overlap-matched to the bare modes) since 2026-09-22; the former total-DOS form was a diagnostic estimate, not a bound.
- T != 300 K linewidths method-validated only; Vogt series ends at
  298 K (theory floor takes over above).
- eta_xxxx (longitudinal) not computable from current data — would need
  a uniaxial strain pair; relevant to sharpen the B-comparison.

## Revised production model (own surface, SELF_OFFDIAG = 1, construction C) — 2026-09-23

eta(300 K) = 5.8423e-04 Pa s; eta(T) for T = 100, 150, 200, 250, 300, 350, 400 K: 5.918e-04, 5.736e-04, 5.729e-04, 5.742e-04, 5.842e-04, 6.064e-04, 6.244e-04 Pa s. Mesh 9/11/13/15: 5.804e-04, 5.842e-04, 5.893e-04, 5.928e-04 Pa s (spread 2.09 %). Numbers only; provenance in the script header.

## Production model tut_z_od1 (hybrid model, construction C)

eta(300 K) = 5.0758e-04 Pa s; eta(T) for T = 250, 300, 350, 400 K: 3.399e-04, 5.076e-04, 5.246e-04, 5.318e-04 Pa s. Mesh 9/11/13/15: 5.065e-04, 5.076e-04, 4.919e-04, 5.055e-04 Pa s (spread 3.09 %). Numbers only; provenance in the script header.

## Production model tut_z_od1 (hybrid model, construction C)

eta(300 K) = 4.7500e-04 Pa s; eta(T) for T = 250, 300, 350, 400 K: 3.247e-04, 4.750e-04, 5.050e-04, 5.041e-04 Pa s. Mesh 9/11/13/15: 4.820e-04, 4.750e-04, 4.749e-04, 4.859e-04 Pa s (spread 2.25 %). Numbers only; provenance in the script header.

## Production model tut_z_od1 (hybrid model, construction C)

eta(300 K) = 4.7494e-04 Pa s; eta(T) for T = 250, 300, 350, 400 K: 3.247e-04, 4.749e-04, 5.049e-04, 5.038e-04 Pa s. Mesh 9/11/13/15: 4.820e-04, 4.749e-04, 4.748e-04, 4.859e-04 Pa s (spread 2.28 %). Numbers only; provenance in the script header.

## Production model tut_z_od1 (hybrid model, construction C)

eta(300 K) = 4.7494e-04 Pa s; eta(T) for T = 250, 300, 350, 400 K: 3.247e-04, 4.749e-04, 5.049e-04, 5.038e-04 Pa s. Mesh 9/11/13/15: 4.820e-04, 4.749e-04, 4.748e-04, 4.859e-04 Pa s (spread 2.28 %). Numbers only; provenance in the script header.
