# lattice-shear-viscosity

Data, code, and figure pipeline for *Mode-resolved lattice shear viscosity
of SrTiO₃ from first-principles anharmonic lattice dynamics* (with a
data-anchored extension to the critical soft sector of BaTiO₃, denoted
$\eta_{44}^{\mathrm{soft}}$). The `latvisc` package performs a mode-resolved evaluation of
the phonon-mode expression

$$
\eta_{ijlm} = \frac{1}{V k_B T} \sum_{\mathbf{q}s} (\hbar\omega)^2\, \gamma_{ij}\, \gamma_{lm}\, n(n+1)\, \tau
$$

($\gamma$ the mode Grüneisen tensor; the lifetime is the exact two-pole
form $\tau = (\Gamma^2 + \omega^2)/(2\Gamma\omega^2)$ from anharmonic
linewidths, which reduces to $1/(2\Gamma)$ for underdamped modes)
using first-principles strain couplings and anharmonic lattice dynamics.
The microscopic inputs — frequencies, strain couplings, and lifetimes —
are computed throughout for SrTiO₃; for BaTiO₃ the soft-mode frequencies
and dampings are taken from measurement. No viscosity parameter is fitted
to acoustic attenuation data; the result is cross-checked against kinetic
theory and, as an order-of-magnitude consistency check, against measured
gigahertz Akhiezer attenuation.

## Layout

| Path               | Contents                                                        |
|--------------------|-----------------------------------------------------------------|
| `dft/qe/`          | Quantum ESPRESSO inputs and small text outputs per material     |
| `dft/azure/`       | VM provisioning and job scripts (credentials via env vars)      |
| `anharmonic/`      | phono3py settings and run scripts per material                  |
| `src/latvisc/`     | Python package: viscosity kernel, Grüneisen, isotope, soft mode|
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
2. Anharmonic lattice-dynamics runs (ALAMODE SCPH-coupled RTA for SrTiO₃;
   inputs and logs in `data/raw/alamode_sto/`, gitignored) produce mode
   linewidths $\Gamma_{\mathbf{q}s}(T)$; the Tamura isotope channel is
   added by `latvisc.isotope`.
3. Strained phonon runs give mode Grüneisen tensors (`latvisc.gruneisen`).
4. `latvisc` computes $\eta(T, f)$ and writes small csv/json to
   `data/processed/` (committed).
5. `scripts/fig*.py` regenerate every paper figure from `data/processed/`
   into `figures/`; each figure maps to exactly one script.

Data sources and copy history are recorded in `PROVENANCE.md`.
