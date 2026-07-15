# Calculation settings audit — dft/qe

Audit of every `scf.in` / `ph.in` under `dft/qe/{SrTiO3,BaTiO3}/`
(2026-07-15). All runs used Quantum ESPRESSO with identical electronic
settings across the two materials.

## Common electronic settings

| Setting | Value | Assessment |
|---|---|---|
| XC functional | PBE (`SLA PW PBX PBC` in all outputs) | See functional note below |
| Pseudopotentials | kjpaw psl 1.0.0 (PAW): Sr/Ba/Ti `pbe-spn`, O `pbe-n` | MD5s in run outputs match `pseudopotentials/SOURCES.md` exactly (Sr `304ac601...`, Ti `7ea84e43...`, O `62c11a45...`, Ba `a94004ef...`) |
| ecutwfc / ecutrho | 60 / 480 Ry | At the audit threshold (>= 60 Ry), ratio 8 appropriate for PAW; header-suggested minima are 37/176 Ry, so comfortably above |
| k-mesh (5-atom cell) | 8x8x8 unshifted | Above the 6x6x6 audit threshold |
| k-mesh (2x2x2 supercell, STO) | 4x4x4 | Equivalent to 8x8x8 on the primitive cell |
| Occupations | fixed, no smearing | Correct for a band insulator |
| SCF conv_thr | 1e-10 (relax), 1e-12 (phonon-preceding scf) | Tight, appropriate for DFPT |
| ph.x tr2_ph | 1e-14 | Tight |
| Born charges / dielectric | `epsil = .true.` in Gamma ph.x runs; 4x4x4 `ldisp` grids rely on the q=0 grid point (see notes in `ph_disp.in`) | LO-TO splitting present in dynmat output (verified: acoustic sum rule applied, LO partners split) |

## Per-material lattice constants

| Material | Relaxed a (this repo) | Experimental a | Deviation | Source of relaxed value |
|---|---|---|---|---|
| SrTiO3 (cubic Pm-3m) | 3.9383 Angstrom (celldm 7.442023 bohr; V = 61.077 Angstrom^3) | 3.905 Angstrom at 300 K [1] | +0.85% | `SrTiO3/relax/vc_relax.out` |
| BaTiO3 (cubic Pm-3m) | 4.0254 Angstrom (celldm 7.606726 bohr; V = 65.222 Angstrom^3) | 4.00 Angstrom just above T_C = 393 K [2] | +0.63% | `BaTiO3/relax/vc_relax.out` |

[1] Okazaki and Kawaminami, Mater. Res. Bull. 8, 545 (1973); the 3.905
Angstrom room-temperature value is standard across neutron/x-ray studies.
[2] Kwei, Lawson, Billinge, and Cheong, J. Phys. Chem. 97, 2368 (1993):
cubic BaTiO3 a = 4.00-4.01 Angstrom in the 400-450 K range.

Both overestimates are the expected PBE behavior (underbinding).

## Functional note (soft-mode relevance)

The entire dataset is PBE. LDA overbinds (smaller cell, weaker or absent
ferroelectric instability); PBE underbinds (larger cell, exaggerated
instability); PBEsol is the usual compromise for perovskite soft modes.
With PBE at the PBE-relaxed volume, the imaginary soft-mode frequencies
below are expected to be on the large side: harmonic |omega| of the
Gamma TO1 instability is 137.8 cm^-1 (SrTiO3) and 244.8 cm^-1 (BaTiO3),
both larger in magnitude than typical LDA/PBEsol literature values. This
does not affect stable-branch validation but must be kept in mind for any
soft-mode-sensitive quantity.

## Calculation inventory

| Directory | Calculation | Key specifics |
|---|---|---|
| `SrTiO3/relax/`, `BaTiO3/relax/` | vc-relax, BFGS, press_conv_thr 0.1 kbar | Converged; scale factors 1.009772427 / 1.004851461 |
| `SrTiO3/harmonic_gamma/`, `BaTiO3/harmonic_gamma/` | ph.x at Gamma with `epsil=.true.` + dynmat.x (`asr='crystal'`, q -> (1,0,0)) | Full Gamma TO/LO inventory |
| `SrTiO3/dispersion/`, `BaTiO3/dispersion/` | ph.x `ldisp` 4x4x4 q-grid; q2r (`zasr='crystal'`), matdyn along Gamma-X-M-Gamma-R-X (40 pts/segment) | 4x4x4 is adequate for band-structure plotting; coarse for full-BZ integrals |
| `SrTiO3/pristine_2x2x2/` | scf input only (40 atoms) | Staged for supercell work; no BaTiO3 counterpart yet |

## Flags

1. **Functional**: PBE, not PBEsol — exaggerated soft-mode instabilities
   and +0.6-0.9% lattice constants are expected and observed. Any future
   re-run campaign should weigh PBEsol; switching later invalidates all
   force constants, so this is a before-Stage-B decision.
2. **DFPT q-grid**: 4x4x4 irreducible set is coarse if the same force
   constants are used to seed anharmonic supercell displacements; fine for
   dispersion plots and instability mapping.
3. No cutoff, k-mesh, smearing, or convergence setting fell below the
   audit thresholds. No inconsistencies between materials.
