# lattice-shear-viscosity

Data, code, and figure pipeline for *Mode-resolved Akhiezer shear viscosity of SrTiO₃ from anharmonic lattice
dynamics: evaluating strain couplings in the renormalised phonon basis* (with a qualitative illustration for the
critical soft sector of cubic BaTiO₃, $\eta_{44}^{\mathrm{soft}}$). The `latvisc` package evaluates the Akhiezer-type
relaxational shear coefficient mode by mode,

$$
\eta_{ijlm} = \frac{1}{V k_B T} \sum_{\mathbf{q}s} (\hbar\omega)^2\, \gamma_{ij}\, \gamma_{lm}\, n(n+1)\, \tau ,
$$

with $\gamma = -\Lambda/(2\omega^2)$ the mode Grüneisen tensor (engineering-shear derivative) and the stress-correlator
lifetime $\tau = 1/(2\Gamma) + 2\Gamma/\omega^2$, which reduces to $1/(2\Gamma)$ for underdamped modes.

SrTiO₃ results are computed for the *hybrid model*: the ALAMODE v1.5.0 example harmonic and anharmonic force constants
(a 2023 refit, external) re-expressed on the 4×4×4 supercell, our own QE-PBEsol shear-strain perturbation, and our own
self-consistent-phonon (SCPH) and strained SCPH solutions, RTA linewidths and projected couplings. The strain coupling
of each renormalised mode is the projection of the strain derivative of the SCPH dynamical matrix onto its eigenvector;
the transferred-coupling and projected-harmonic constructions are kept for comparison. The room-temperature value at the
finest completed SCPH inner mesh (12³) is a finite-mesh model result; its inner-mesh convergence is not established (see
the paper, Sec. 4.3). No parameter is fitted to acoustic data. For BaTiO₃ the soft-mode frequencies and dampings are
measured values.

## Version and archive

This is release **v3.0** (git tag `v3.0`), the version of the revised manuscript. It is archived on Zenodo:

- version DOI: 10.5281/zenodo.22983132
- concept DOI (always resolves to the latest version): 10.5281/zenodo.21940876

The Zenodo record holds two files: the repository at tag v3.0 (`lattice-shear-viscosity-v3.0.zip`) and the raw SCPH/RTA
runs (`lattice-shear-viscosity-v3.0-raw.zip`, contents in `data/raw/MANIFEST.md`).

## Layout

| Path                          | Contents |
|-------------------------------|----------|
| `dft/qe/`                     | Quantum ESPRESSO inputs and small text outputs per material, including the shear-strained cells |
| `dft/azure/`                  | VM provisioning and job scripts (credentials via environment variables) |
| `anharmonic/SrTiO3/alamode_tutorial_inputs/` | external ALAMODE v1.5.0 example force constants, with licence and checksums |
| `src/latvisc/`                | Python package: viscosity kernel, Grüneisen tensor, projected couplings, isotope rate, soft mode |
| `scripts/`                    | production assembly, tables and one script per paper figure |
| `scripts/scph/`               | hybrid-model construction and the SCPH driver; rebuild sequence in `scripts/scph/README.md` |
| `scripts/v3_diagnostics/`     | scripts of the diagnostic tables cited in the paper and Supplement |
| `data/processed/`             | small csv/json derived data (committed) |
| `data/processed/v3_diagnostics/` | diagnostic tables; `README.md` there maps each file to its script and to the paper section |
| `data/raw/`                   | SCPH/RTA runs (not in git; Zenodo raw archive; `MANIFEST.md` is committed) |
| `figures/`                    | generated figures (image files are gitignored; regenerate with the scripts) |
| `tests/`                      | units, high-T limit, kinetic cross-check, strain convention, projection, acoustic limit, isotope |
| `references/digitized/`       | digitised literature data with per-figure READMEs |

## Quickstart

```bash
uv sync
uv run pytest
```

## Reproduction workflow

1. QE relax and DFPT runs (`dft/azure/` provisioning); inputs and small text outputs in `dft/qe/<material>/`,
   including the shear-strained cells ε_xy = ±0.005, ±0.010.
2. `scripts/scph/build_hybrid_model.py` re-expresses the example harmonic set on the 4×4×4 supercell and builds the
   strained sets D_tut ± h K_QE (ALAMODE 1.5.0 built locally; work directory in `DYNMAT_SCRATCH`).
3. `scripts/scph/run_scph_surface.py` solves SCPH (SELF_OFFDIAG = 1) for the unstrained and strained sets at the chosen
   correction/inner meshes and runs the 8³ RTA; outputs in `data/raw/alamode_sto/`. To skip steps 1–3, unpack the raw
   archive at the repository root (see `data/raw/MANIFEST.md`).
4. `scripts/compute_eta_SrTiO3.py` (production), `compute_eta_isotope_SrTiO3.py`, `compute_eta_BaTiO3.py`,
   `table1_convergence.py`, `table_sensitivity.py`, `export_coupling_constructions.py`, `export_hybrid_dispersion.py`
   and the audit scripts write small csv files to `data/processed/`; `scripts/v3_diagnostics/` writes the diagnostic
   tables to `data/processed/v3_diagnostics/`.
5. Figures, one script each, from `data/processed/` into `figures/`:

   | figure | script |
   |---|---|
   | Fig. 1 | `scripts/fig1_composite.py` |
   | Fig. 2 | `scripts/fig2_coupling_constructions.py` |
   | Fig. 3 | `scripts/fig3_eta_T.py` |
   | Fig. 4 | `scripts/fig4_isotope.py` |
   | Fig. 5 | `scripts/fig5_eta_BaTiO3_sector.py` |
   | graphical abstract | `scripts/graphical_abstract.py` |

   Scripts prefixed `extra_` make figures that are not in the paper.

Data sources and copy history are recorded in `PROVENANCE.md`.
