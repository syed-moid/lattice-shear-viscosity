# Pseudopotential provenance

Source: pslibrary suggested-PP table — https://dalcorso.github.io/pslibrary/PP_list.html

The `.UPF` files themselves are NOT committed (see `.gitignore`); this file is
the committed record of exactly which files every run used. Verify after any
re-download with `md5 -r dft/qe/pseudopotentials/*.UPF` (or `md5sum` on Linux).

MD5s for Sr/Ti/O below were cross-checked 2026-07-04 two ways: computed
locally from the files, AND matched against the `MD5 check sum` lines QE
printed in `vc_relax.out` and `scf_phonon_quality.out` (the actual production
runs). They agree exactly.

| Element | Filename                          | MD5                                | Verified |
|---------|-----------------------------------|------------------------------------|----------|
| Sr      | Sr.pbe-spn-kjpaw_psl.1.0.0.UPF    | `304ac60131d69ebf2380f6d2a1d0bd9f` | local file + vc_relax.out + scf_phonon_quality.out |
| Ti      | Ti.pbe-spn-kjpaw_psl.1.0.0.UPF    | `7ea84e4330344cbc938d542ff58a5c10` | local file + vc_relax.out + scf_phonon_quality.out |
| O       | O.pbe-n-kjpaw_psl.1.0.0.UPF       | `62c11a45e085f03adfe5482d7843b034` | local file + vc_relax.out + scf_phonon_quality.out |
| Ba      | Ba.pbe-spn-kjpaw_psl.1.0.0.UPF    | `a94004ef8cdd1ef6024fa921b83633cc` | local file; source confirmed 2026-07-06: https://pseudopotentials.quantum-espresso.org/upf_files/Ba.pbe-spn-kjpaw_psl.1.0.0.UPF — run cross-check CONFIRMED 2026-07-06: BTO vc_relax.out banner prints the same MD5. Header suggests ecutwfc ≥ 37 Ry / ecutrho ≥ 176 Ry; we run 60/480 (STO-consistent), comfortably above. |
