# Diagnostic tables of the v3.0 paper

These tables support statements in the paper *Mode-resolved Akhiezer shear viscosity of SrTiO₃ from anharmonic lattice
dynamics: evaluating strain couplings in the renormalised phonon basis* (main text "Sec.", Supplementary Material "S")
that are not already covered by the production tables in `data/processed/`. The producing scripts are in
`scripts/v3_diagnostics/` and are run from the repository root, e.g. `uv run python scripts/v3_diagnostics/<name>.py`;
they write into this folder.

Terminology. *Hybrid model*: the production model (example harmonic set re-expressed on the 4×4×4 supercell plus the
QE-PBEsol strain perturbation; correction mesh 2³, inner mesh 12³). *Diagnostic surface*: the same procedure with the
own QE-PBEsol harmonic set. Mesh tags `iNsM` = correction mesh N³ (ALAMODE `KMESH_INTERPOLATE`), inner mesh M³
(`KMESH_SCPH`). Constructions: A transferred coupling, B projected harmonic coupling, C projected SCPH coupling.

Reproducibility status. **R** = regenerated exactly by the listed script from the repository plus the raw archive
(`data/raw/`, see `data/raw/MANIFEST.md`). **R\*** = needs the ALAMODE work directory as well (`DYNMAT_SCRATCH`, an
ALAMODE 1.5.0 build; the scripts call anphon). **Record** = a table of the submitted (v2.2) model computed with the
legacy assembly; kept as the record of the statement it supports; the current code no longer reproduces it bit for bit
(the legacy assembly has since changed its default lifetime kernel).

| file | contents | script | status | cited in |
|---|---|---|---|---|
| `mode_table_SrTiO3_300K_v6.csv` | per-mode table of the production η(300 K): ω, partner ω₀, Λ_A/Λ_B/Λ_C, γ, Γ, τ, Γ_iso, contribution | `scripts/compute_eta_SrTiO3.py` + `scripts/compute_eta_isotope_SrTiO3.py` | R | Secs. 4.1, 4.5, 6.2; Table 7 |
| `narrative_facts_v6_bins.csv` | η_C and η_A by partner-ω₀ and ω_r bin; η-weighted ⟨γ²⟩, ⟨τ⟩ per bin | `narrative_facts.py` | R | Sec. 4.1, Sec. 6.2 |
| `narrative_facts_300K.txt` | scale numbers (kinetic estimate, same-inputs reference, soft sectors), GHz conversions, isotope shares | `narrative_facts.py` | R | Secs. 2.4, 4.5, 4.6 (Table 5), 6.2 |
| `budget_numbers_300K.txt` | finite-frequency factors, coherence exposure, derivative-noise bias, kernel ratio, Γ-triplet transfer indicator, soft-branch taper | `budget_numbers.py` | R | Table 7 (groups B, D, E) |
| `z_kmesh_convergence.csv` | hybrid model per inner mesh 2³–16³: η_C (consistent, fixed linewidths), frequencies, imaginary counts, κ, τ medians, R-shell share, noise/signal, strain-step statistics | `eta_z.py` (per mesh, then) `z_kmesh_convergence.py` | R | Table 4, Sec. 4.3 |
| `eta_z_i2s12_outer_zone.csv`, `eta_z_i2s8_outer_zone.csv` | η_A/B/C on outer meshes 9³–15³ at inner mesh 12³ and 8³ | `eta_z.py <mesh> --outer` | R | Table 2, Sec. 4.3 |
| `z_check_11p1iii.csv` | anphon SCPH frequencies of the re-expressed vs original example harmonic set (2³/2³, 2³/8³) | `z_check_11p1iii.py` | R\* | Sec. 3.2, S4 |
| `k5b_indicator_11p3.csv` | eigenbasis-sensitivity indicator: the same K_QE projected on the tutorial vs QE harmonic eigenvectors, totals and per bin | `k5b_indicator_11p3.py` | R | Table 7 (B), Sec. 5 |
| `benchmark_variants_11p0.csv` | example set under our settings: SCPH frequencies for SELF_OFFDIAG 0/1 × NONANALYTIC 0/2/3 at 2³/8³ and 2³/12³ | `benchmark_variants_11p0.py` | R\* | Sec. 3.3, S1 (Table S2) |
| `benchmark_kappa_11p0.csv` | κ(300 K) of the example set at 2³/8³, SELF_OFFDIAG 0/1, 8³ and 12³ Boltzmann meshes | `benchmark_variants_11p0.py` | R\* | Sec. 3.5, S1 |
| `tutorial_kmesh_control_9p4.csv`, `tutorial_kmesh_control_9p4_interp2.csv` | example set at correction mesh 4³ (sum-rule artefact) and 2³ (Tadano–Tsuneyuki protocol), inner mesh 2³–16³ | `tutorial_kmesh_control.py` (argument `2/4 2/8 2/12 2/16 --suffix interp2` for the second) | R\* | S1 (Table S1), S2 |
| `refit_provenance.md` | what the parameter counts of the 2015 fit and the 2023 example refit count | written by hand from the ALAMODE fit logs | — | Sec. 3.3, S1 |
| `kmesh_convergence_9p1.csv` | diagnostic surface per mesh (2³/2³ … 4³/12³, 2³/16³): η_C consistent/fixed, frequencies, κ, τ, noise/signal | `eta_constructions.py` (per mesh) then `kmesh_convergence_9p1.py` | R | Sec. 3.6, S2 (Table S3), Table 7 |
| `kmesh_interp_9p2.csv` | correction-mesh sensitivity at fixed inner mesh; memory estimates of the runs not done | `kmesh_interp_9p2.py` | R | S2, S4, Sec. 4.3 |
| `T_stability_9p6.csv` | diagnostic surface: stability and convergence vs temperature at 2³/2³, 4³/8³, 4³/12³ | `T_stability_9p6.py` | R | Sec. 3.6, S2 |
| `loop_weight_estimate_9p5.csv` | share of the loop weight Σ(2n′+1)/(2ω′) on each surface's own converged ω′ at uniform vertex (a weight estimate, not the quartic contraction) | `loop_weight_estimate_9p5.py` | R | S2 |
| `freeze_6p5b1.csv` | diagnostic surface at 2³/2³, 4³/4³, 4³/8³: iterations, residuals, R-point, κ, τ, R-region share | `freeze_6p5b1.py` | R | S2 |
| `step_assessment_6p5b2_od1_i2s2.csv` | diagnostic surface: strain-step (h = 0.010 vs 0.020) classes and Richardson variants | `step_assessment_6p5b2.py` | R | S4 |
| `eta_constructions_b5_zone.csv`, `eta_constructions_sym_zone.csv` | diagnostic surface η_A/B/C with the production and with symmetrised (even offset removed) strained inputs | `eta_constructions.py`; inputs from `make_symmetrised_inputs.py` | R\* | Table 7 (A), S4 |
| `na_treatment_6p5b4.csv` | odd non-analytic part (rigid-ion vs Ewald) and projected vs eigenvalue-matched couplings, per mode | `na_treatment_6p5b4.py` | R | Sec. 3.4, Appendix B, Table 7, S4 |
| `coupling_constructions_selected.csv`, `coupling_constructions_allmodes.csv`, `acoustic_limit_6p3c.csv`, `selection_6p3a.csv` | constructions A/B/C and the matched-route comparison on selected high-weight slots of the diagnostic surface; acoustic limit on the diagnostic surface; the selection | `select_modes.py`, `coupling_selected.py` | Record (the selection used the v4 mode table of the submitted analysis) | Sec. 3.4 (median 1.3 %), Appendix B, S3 |
| `kernel_quantum_check.csv`, `kernel_classical_check.csv` | quantum and classical checks of the two lifetime kernels (modes and synthetic cases) | `overdamped_kernel_check.py` | Record (mode rows); synthetic rows R | Sec. 2.6, Table 7 |
| `eta_SrTiO3_kernel_comparison.csv` | submitted-model SrTiO₃ η with energy vs stress kernel | `recompute_eta_stress_kernel.py` | Record | Sec. 2.6, Table 7 |
| `eta_BaTiO3_kernel_comparison.csv` | BaTiO₃ soft-sector η with energy vs stress kernel, 410–700 K (tensor-convention coupling) | `recompute_eta_stress_kernel.py` | R | Sec. 4.7, Table 7 (D) |
| `bto_stable_manifold_estimate.csv` | order-of-magnitude estimate of the BaTiO₃ stable manifold relative to the soft sector | `bto_stable_manifold_estimate.py` | Record | Table 7 (D) |
| `isotope_series_exact_projection.csv` | ¹⁸O series with the density-of-states form and the exact site projection (submitted model) | `isotope_exact_projection.py` | Record | Sec. 2.5, Sec. 4.5 |
| `thermoelastic_estimate_300K.csv` | thermoelastic damping of 66–74 GHz longitudinal waves | `thermoelastic_estimate.py` | R | Sec. 4.6 (Table 5) |
| `energy_projection_check_300K.csv` | projection of the stress onto the conserved energy | `energy_projection_check.py` | R | Sec. 2.2, Table 7 |
| `shear_normalization_toy.csv` | analytic shear deformation: path derivative vs tensor derivative (the factor 4 in η) | `check_shear_normalization.py` | R | Sec. 2.1 |
| `mode_table_SrTiO3_300K.csv` | per-mode table of the submitted model (input of the kernel, energy-projection and isotope records) | `mode_table_300K.py` (needs `data/raw/gruneisen_modes/`) | Record | — (input) |

Validation tables of the hybrid-model construction are in `data/processed/reports/hybrid_model_reexpression_validation.csv`
and `hybrid_model_strained_validation.csv` (script `scripts/scph/build_hybrid_model.py`).
