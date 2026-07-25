# Pseudopotential provenance

Source: pslibrary suggested-PP table — https://dalcorso.github.io/pslibrary/PP_list.html

The `.UPF` files themselves are NOT committed (see `.gitignore`); this file is
the committed record of exactly which files every run used. Verify after any
re-download with `md5 -r dft/qe/pseudopotentials/*.UPF` (or `md5sum` on Linux).

MD5s for Sr/Ti/O below were cross-checked 2026-07-04 two ways: computed
locally from the files, AND matched against the `MD5 check sum` lines QE
printed in `vc_relax.out` and `scf_phonon_quality.out` (the actual production
runs). They agree exactly.

## PBEsol set (Gate G1 resolution: pipeline functional switch, 2026-07-15)

Downloaded 2026-07-15 from
https://pseudopotentials.quantum-espresso.org/upf_files/ (same psl 1.0.0
kjpaw family as the PBE set; `functional` header field `SLA PW PSX PSC`
verified for Sr). `.UPF` files stay gitignored.

| Element | Filename | MD5 | Source |
|---------|----------|-----|--------|
| Sr | Sr.pbesol-spn-kjpaw_psl.1.0.0.UPF | `5f6d651ace59fc2f5ab95f6a3abc31cb` | QE upf_files |
| Ti | Ti.pbesol-spn-kjpaw_psl.1.0.0.UPF | `37a32a5821746b2d50d64995a8e5b789` | QE upf_files |
| O  | O.pbesol-n-kjpaw_psl.1.0.0.UPF    | `cb766521a97cf798d01896eaf7ac9a0a` | QE upf_files |
| Ba | Ba.pbesol-spn-kjpaw_psl.1.0.0.UPF | `dd46dd63df357211ac98124515369fe1` | not hosted on the QE site (404 as of 2026-07-15); generated 2026-07-15 with ld1.x (QE 6.7MaX, Azure VM) from the pslibrary recipe `Ba.pbesol-spn-kjpaw_psl.1.0.0.in` in this directory (extracted from `paw_ps_high.job`, github.com/dalcorso/pslibrary master, GPL; `dft='PBESOL'`, `rel=1` matching the scalar-relativistic PBE set). Post-processing: three over-long data lines (unwrapped PP_DIJ/augmentation rows emitted by ld1.x 6.7, up to 6370 chars) were rewrapped to <= 4 values/line because pw.x's XML reader rejects lines > 1024 chars; numeric content unchanged. The MD5 above is the rewrapped (canonical) file; generation log and unmodified original in `data/raw/ld1_Ba_pbesol/` (gitignored). Cross-check this MD5 against the pw.x banner of every run that uses it. |

## PBE set (original audit reference)

| Element | Filename                          | MD5                                | Verified |
|---------|-----------------------------------|------------------------------------|----------|
| Sr      | Sr.pbe-spn-kjpaw_psl.1.0.0.UPF    | `304ac60131d69ebf2380f6d2a1d0bd9f` | local file + vc_relax.out + scf_phonon_quality.out |
| Ti      | Ti.pbe-spn-kjpaw_psl.1.0.0.UPF    | `7ea84e4330344cbc938d542ff58a5c10` | local file + vc_relax.out + scf_phonon_quality.out |
| O       | O.pbe-n-kjpaw_psl.1.0.0.UPF       | `62c11a45e085f03adfe5482d7843b034` | local file + vc_relax.out + scf_phonon_quality.out |
| Ba      | Ba.pbe-spn-kjpaw_psl.1.0.0.UPF    | `a94004ef8cdd1ef6024fa921b83633cc` | local file; source confirmed 2026-07-06: https://pseudopotentials.quantum-espresso.org/upf_files/Ba.pbe-spn-kjpaw_psl.1.0.0.UPF — run cross-check CONFIRMED 2026-07-06: BTO vc_relax.out banner prints the same MD5. Header suggests ecutwfc ≥ 37 Ry / ecutrho ≥ 176 Ry; we run 60/480 (STO-consistent), comfortably above. |
