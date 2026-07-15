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
