# Provenance — dft/qe/SrTiO3

All files copied 2026-07-15 from the ferroelectric-ins-ml project
(the author's local `ferroelectric-ins-ml` working copy).

| Directory          | Origin                          | Contents                                                        |
|--------------------|---------------------------------|-----------------------------------------------------------------|
| `relax/`           | `qe/SrTiO3/relax/`              | vc-relax input and output (cubic 5-atom cell)                   |
| `harmonic_gamma/`  | `qe/SrTiO3/harmonic_gamma/`     | Γ-point ph.x run, dynmat post-processing, run logs              |
| `dispersion/`      | `qe/SrTiO3/dispersion/`         | ph.x dispersion (`SrTiO3.dyn*`), q2r, matdyn, `SrTiO3.444.fc`   |
| `pristine_2x2x2/`  | `qe/SrTiO3/pristine_2x2x2/`     | 2x2x2 supercell scf input                                       |

No files exceeded the 5 MB copy limit; nothing was skipped except
`.DS_Store` and `.gitkeep` placeholders.

Not copied (kept in the origin repo): `qe/anchoring/vogt1995_fig6/` —
digitized soft-mode frequency and damping data for SrTiO₃ from
Vogt, Phys. Rev. B 51, 8046 (1995), Fig. 6. Re-copy into
`data/processed/` if needed for the soft-mode figure.

## Force-constant file replacement (2026-07-16)

The `dispersion/SrTiO3.444.fc` copied from the origin project proved
inconsistent with the `.dyn` set and `.freq` committed alongside it:
matdyn.x runs from that fc did not reproduce the committed Gamma
frequencies (uniform-mesh audit, `dft/qe/mesh_audit/`). The fc was
regenerated with q2r.x (zasr='crystal', QE 6.7) from the committed
`SrTiO3.dyn0..dyn10`; the regenerated fc reproduces the committed
band-path frequencies to within 0.06 cm⁻¹ along the whole path (exact at Gamma). The
stale original is preserved in `data/raw/stale_pbe_fc/` (gitignored).
All committed PBEsol fc files were verified self-consistent the same way.
