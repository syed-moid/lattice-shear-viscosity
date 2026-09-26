# Provenance

## DFT data (`dft/qe/`)

Quantum ESPRESSO inputs and small text outputs copied 2026-07-15 from the
local ferroelectric-ins-ml project. Per-material details in
`dft/qe/SrTiO3/PROVENANCE.md` and `dft/qe/BaTiO3/PROVENANCE.md`;
pseudopotential sources and MD5s in `dft/qe/pseudopotentials/SOURCES.md`
(the `.UPF` files themselves are not committed).

## Azure pipeline (`dft/azure/`)

`provision_vm.py`, `run_job.py`, `teardown_vm.py` copied from the same
project. All credentials and subscription identifiers load from environment
variables (template: `dft/azure/.env.example`); nothing sensitive is stored
in the repository.

## Earlier viscosity scripts

An earlier analysis codebase (local, unpublished) was reviewed during setup.
Its viscosity expressions were superseded by the formulation implemented in
`src/latvisc/viscosity.py` and none of its equations, coefficients, or
fitted constants were carried over. Items adapted from it:

- Overflow-safe evaluation pattern for Bose-Einstein factors
  (`src/latvisc/viscosity.py`).
- Literature transition temperatures (T_C values) recorded in
  `src/latvisc/materials.py`; lattice constants and cell volumes there come
  from this repository's own vc-relax outputs, not from the old scripts.

## External force constants (`anharmonic/SrTiO3/alamode_tutorial_inputs/`)

The harmonic (`STO222.xml`) and anharmonic (`STO_anharm.xml.bz2`, orders 3–6) force constants and `BORN` are the SrTiO₃
example of ALAMODE (https://github.com/ttadano/alamode, tag `v.1.5.0`, commit d86ab0f1), redistributed under its MIT
licence (`ALAMODE_LICENSE.txt`); checksums in `SHA256SUMS.txt`. The fit logs date these files to April 2023, i.e. a refit
with ALM 1.5.0, not necessarily the force constants of the 2015 SCPH study; details and parameter counts in
`data/processed/v3_diagnostics/refit_provenance.md`.

## Hybrid model and SCPH runs (v3.0)

The production (hybrid) model combines the example harmonic set, re-expressed on the 4×4×4 supercell, with the shear-strain
perturbation of this work's QE-PBEsol force constants (`dft/qe/SrTiO3/dispersion_pbesol/`, `gruneisen_pbesol/`). All SCPH,
dfc2 and RTA runs were made locally with ALAMODE 1.5.0 by `scripts/scph/`; they are in the raw archive
(`data/raw/MANIFEST.md`), not in git. Diagnostic tables of the paper and Supplement, with their producing scripts, are listed
in `data/processed/v3_diagnostics/README.md`; tables marked "record" there belong to the submitted model and are kept as
the record of the statements they support.

## Archive

Release v3.0 is archived on Zenodo (version DOI 10.5281/zenodo.[VERSION DOI — inserted after deposit]; concept DOI
10.5281/zenodo.21940877): the repository at tag v3.0 and the raw-run archive with `MANIFEST.md` and `SHA256SUMS.txt`.
