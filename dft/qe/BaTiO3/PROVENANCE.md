# Provenance — dft/qe/BaTiO3

All files copied 2026-07-15 from the ferroelectric-ins-ml project
(the author's local `ferroelectric-ins-ml` working copy).

| Directory          | Origin                          | Contents                                                        |
|--------------------|---------------------------------|-----------------------------------------------------------------|
| `relax/`           | `qe/BaTiO3/relax/`              | vc-relax input and output (cubic 5-atom cell)                   |
| `harmonic_gamma/`  | `qe/BaTiO3/harmonic_gamma/`     | Γ-point ph.x run, dynmat post-processing, run logs              |
| `dispersion/`      | `qe/BaTiO3/dispersion/`         | ph.x dispersion (`BaTiO3.dyn*`), q2r, matdyn, `BaTiO3.444.fc`   |

No files exceeded the 5 MB copy limit; nothing was skipped except
`.DS_Store` and `.gitkeep` placeholders. The origin `qe/BaTiO3/scf/`,
`qe/BaTiO3/ph/`, and `qe/BaTiO3/phono3py/` directories contained only
placeholders and were not copied. No BaTiO₃ supercell (pristine 2x2x2)
run exists yet; supercell and phono3py runs are pending.

## Force-constant file replacement (2026-07-16)

The `dispersion/BaTiO3.444.fc` copied from the origin project proved
inconsistent with the `.dyn` set and `.freq` committed alongside it:
matdyn.x runs from that fc did not reproduce the committed Gamma
frequencies (uniform-mesh audit, `dft/qe/mesh_audit/`). The fc was
regenerated with q2r.x (zasr='crystal', QE 6.7) from the committed
`BaTiO3.dyn0..dyn10`; the regenerated fc reproduces the committed
band-path frequencies to within 0.06 cm⁻¹ along the whole path (exact at Gamma). The
stale original is preserved in `data/raw/stale_pbe_fc/` (gitignored).
All committed PBEsol fc files were verified self-consistent the same way.
