# lattice-shear-viscosity

Data, code, and figure pipeline for *Lattice shear viscosity in strongly
anharmonic ferroelectric oxides from first principles* (SrTiO3 and BaTiO3).
The `latvisc` package evaluates the phonon-mode expression

$$
\eta_{ijlm} = \frac{1}{V k_B T} \sum_{\mathbf{q}s} (\hbar\omega)^2\, \gamma_{ij}\, \gamma_{lm}\, n(n+1)\, \tau
$$

($\gamma$ the mode Grueneisen tensor, $\tau = 1/(2\Gamma)$ from anharmonic
linewidths, with an overdamped-safe effective lifetime for the soft mode)
from first-principles phonon frequencies, linewidths, and mode Grueneisen
parameters, and cross-checks the result against kinetic theory and Akhiezer
attenuation.

## Layout

| Path               | Contents                                                        |
|--------------------|-----------------------------------------------------------------|
| `dft/qe/`          | Quantum ESPRESSO inputs and small text outputs per material     |
| `dft/azure/`       | VM provisioning and job scripts (credentials via env vars)      |
| `anharmonic/`      | phono3py settings and run scripts per material                  |
| `src/latvisc/`     | Python package: viscosity kernel, Grueneisen, isotope, soft mode|
| `data/raw/`        | Large phono3py hdf5 outputs (gitignored)                        |
| `data/processed/`  | Small csv/json derived data (committed)                         |
| `scripts/`         | One script per paper figure and table                           |
| `figures/`         | Generated figures (committed)                                   |
| `tests/`           | Units check, high-T limit, kinetic-theory cross-check           |

## Quickstart

```bash
uv sync
uv run pytest
```

## Reproduction workflow

1. QE relax and DFPT runs on Azure VMs (`dft/azure/`); inputs and small
   text outputs land in `dft/qe/<material>/`.
2. phono3py runs locally (`anharmonic/<material>/`) produce mode linewidths
   $\Gamma_{\mathbf{q}s}(T, f)$ as hdf5 in `data/raw/` (gitignored).
3. Strained phonon runs give mode Grueneisen tensors (`latvisc.gruneisen`).
4. `latvisc` computes $\eta(T, f)$ and writes small csv/json to
   `data/processed/` (committed).
5. `scripts/fig*.py` regenerate every paper figure from `data/processed/`
   into `figures/`; each figure maps to exactly one script.

Data sources and copy history are recorded in `PROVENANCE.md`.
