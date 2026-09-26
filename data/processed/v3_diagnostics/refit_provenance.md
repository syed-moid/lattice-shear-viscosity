# Provenance of the ALAMODE example force constants and what the parameter counts count

Source: https://github.com/ttadano/alamode, release tag `v.1.5.0`, commit d86ab0f1335153809306fb592d42977bbebb2545 (tag date
2024-02-29; release 1.5.0 dated 2023-03-31 in `ChangeLog.md`), directory `example/SrTiO3/reference/`. The same files, with
their licence and checksums, are in `anharmonic/SrTiO3/alamode_tutorial_inputs/`.

| file | SHA-256 | content |
|---|---|---|
| `STO222.xml` | 9739c36af71eb1ab306760e0a86a7ba09f7337c6220901272fea1b20400f9f0f | harmonic force constants, 2×2×2 supercell (40 atoms), ordinary least squares (`ALM1.in`, `DFSET_harmonic`); `<ALM_version>1.5.0` |
| `STO_anharm.xml.bz2` | 4bdc3b592a9f22018c632645ea39313653688c94cba18ba0e8f8baee29a46485 | harmonic + orders 3–6 (elastic net, `opt.in`); decompressed `STO_anharm.xml` 5e787b95eb59fdc2a5cd93a3fe298d976f71d3a83905dce568cc021d86e71b79 |
| `BORN` | e283fbf4a6d75c7dbe00a8d8589c1a0a241ea9a4255bc1fc1b078e2c229a1bb9 | ε∞ and Born charges of the example cell |

Dates: `ALM1.log` "Job started at Sun Apr 2 02:45:01 2023", `opt.log` "Sun Apr 2 02:49:05 2023". The distributed files are
therefore a 2023 refit produced with ALM 1.5.0 — not necessarily the force constants underlying the 2015 results of Tadano and
Tsuneyuki.

## What the counts count

| quantity | 2023 example (`opt.log`) | 2015 paper (Tadano & Tsuneyuki, Sec. IV A) |
|---|---|---|
| orders fitted | 3, 4, 5, 6 (harmonic fixed from `STO222.xml`) | 3, 4, 5, 6 (harmonic fixed from finite displacement) |
| interaction range | cutoff 12 bohr for orders 3–6; NBODY = 2 3 3 2 2 | cubic: all within the 2×2×2 supercell; quartic: up to third-nearest-neighbour shells; 5th/6th: nearest-neighbour pairs |
| symmetry-distinct terms before constraints | 6851 = 49 (harmonic) + 746 + 4527 + 386 + 1143 | not stated |
| **independent anharmonic parameters after space-group and translational-invariance constraints** | **3081** = 698 + 2215 + 43 + 125 | **1053** |
| non-zero after the sparse fit | 1990 = 530 + 1340 + 37 + 83 (residual 2.30 %) | not stated |
| method | elastic net, L1_ALPHA = 3.4548×10⁻⁶ (fixed) | LASSO (split Bregman), λ from four-fold cross validation |
| training data | 40 configurations (`DFSET_AIMD+random`) | 40 configurations, AIMD 500 K + 0.1 Å random displacements |

The comparable pair is 3081 vs 1053 independent anharmonic parameters (same definition in both sources); 6851 counts a
different object and is not compared with 1053. The 2023 basis is about three times larger, mainly in the quartic order. This
supports, but does not demonstrate, the statement in the paper that the refit is a plausible, unresolved source of the
25 vs 35 cm⁻¹ difference of the R-point SCPH frequency.
