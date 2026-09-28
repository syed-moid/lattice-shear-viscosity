# data/raw — manifest

`data/raw/` holds the heavy calculation outputs. It is not in the git repository (only this manifest is). The folders marked
**archived** are distributed as two raw-data archives on Zenodo (DOIs: see `README.md`): `lattice-shear-viscosity-v3.0-raw.zip`
(release v3.0) and `lattice-shear-viscosity-v3.1-raw.zip` (release v3.1: the raw files new or changed since v3.0), each with
this manifest and `SHA256SUMS.txt` inside. Unzip the v3.0 archive and then the v3.1 archive at the repository root so that
the paths below resolve.

Not in the archive: pseudopotential (UPF) files, superseded runs, and the `STO_anharm.xml` symbolic links. Every SCPH/RTA folder
contains a link `STO_anharm.xml` to the decompressed ALAMODE example quartic set; recreate them after unpacking:

```bash
bunzip2 -k anharmonic/SrTiO3/alamode_tutorial_inputs/STO_anharm.xml.bz2
A=$(pwd)/anharmonic/SrTiO3/alamode_tutorial_inputs/STO_anharm.xml
find data/raw/alamode_sto -mindepth 1 -type d -exec sh -c 'ls "$1"/*.in >/dev/null 2>&1 && ln -sf "$2" "$1/STO_anharm.xml"' _ {} "$A" \;
```

Mesh tags `iNsM`: SCPH correction mesh N³ (`KMESH_INTERPOLATE`), inner mesh M³ (`KMESH_SCPH`). `od1` = SELF_OFFDIAG = 1. Set
names: `reference` (unstrained), `shear_xy_p005`/`m005` (engineering shear ±0.005, h = 0.010), `p010`/`m010` (±0.010, h = 0.020);
prefix `z_` = hybrid model. File types per run: `renorm_*.xml` (dfc2 renormalised FC2), `mesh11_*.{in,log,npz}` (frequencies and
eigenvectors on the 11³ outer mesh), `STO_RTA_*.{in,log,result,kl}` (8³ RTA linewidths), `scph_*`/`dfc2_*` logs, `runs.csv` (timing,
iterations, final residual).

| path | size | contents | produced by | archive |
|---|---|---|---|---|
| `alamode_sto/z_tut/` (top level) | 7 M | hybrid model sets: `STO444_tut.xml` (example harmonic set on 4×4×4), strained `z_shear_xy_*_full_fc2.xml`, BORN files, q2r-format `z_*.fc` | `scripts/scph/build_hybrid_model.py reexpress|strain|qefc` | archived |
| `alamode_sto/z_tut/i2s{2,4,8,12,16}/` | 121 M | hybrid model SCPH (unstrained + strained) and RTA at 300 K; `i2s12` is the production mesh | `scripts/scph/run_scph_surface.py` | archived |
| `alamode_sto/z_tut/i2s12_T/` | 65 M | hybrid model at 2³/12³ for 200, 250, 350, 400 K (200 K strained set not converged, kept with suffix `.unconverged` where present) | `run_scph_surface.py --label i2s12T<T>` | archived |
| `alamode_sto/z_tut/i2s12_rta_mesh/` | 5 M | RTA at 300 K on the production renormalised set (2³/12³) and the bare set with scattering meshes 10³ and 12³ (sensitivity) | anphon `MODE = RTA`; `scripts/v3_diagnostics/scattering_mesh_sensitivity.py` | archived (v3.1 increment) |
| `alamode_sto/z_tut/bare/` | 1.3 M | harmonic (non-SCPH) RTA of the example set; linewidth reference for the projected-harmonic construction | ALAMODE anphon RTA | archived |
| `alamode_sto/own_od1/i2s2, i2s4, i2s8, i2s12, i2s16, i3s12, i4s4, i4s8, i4s12, i4s16` | 305 M | diagnostic surface (own QE-PBEsol harmonic set), SCPH + RTA at 300 K per mesh | `run_scph_surface.py` | archived |
| `alamode_sto/own_od1/i2s2_sym, i2s2_sym_eval` | 15 M | diagnostic surface with symmetrised strained inputs; `i2s2_sym_eval` holds relative links to the evaluated set | `scripts/v3_diagnostics/make_symmetrised_inputs.py` + `run_scph_surface.py` | archived |
| `alamode_sto/own_od1/i4s8_Tstab, i4s12_Tstab` | 27 M | diagnostic surface vs temperature | `run_scph_surface.py --temps ...` | archived |
| `alamode_sto/own_od1/driver_*.log` | < 1 M | driver logs | `run_scph_surface.py` | archived |
| `alamode_sto/own_surface_6p2/` | 67 M | diagnostic surface at 2³/2³, SELF_OFFDIAG 0 and 1, with the ±0.010 pair; input of the coupling-construction comparison and the acoustic-limit table | `run_scph_surface.py` (earlier layout) | archived |
| `alamode_sto/own_surface_i2s2/` | 67 M | diagnostic surface 2³/2³ with the distributed sum rule (reference for the strain-step assessment) | `run_scph_surface.py` | archived |
| `alamode_sto/surface_tutorial/` | 50 M | example set SCPH at 100–400 K on 11³ (submitted-model surface; benchmark) | ALAMODE anphon | archived |
| `alamode_sto/example_set_benchmark/bench11/` | 12 M | example set under our settings: SELF_OFFDIAG × NONANALYTIC variants and κ | `scripts/v3_diagnostics/benchmark_variants_11p0.py` | archived |
| `alamode_sto/example_set_benchmark/tutorial_kmesh/` | 16 M | example set, correction mesh 2³ and 4³, inner mesh 2³–16³ | `scripts/v3_diagnostics/tutorial_kmesh_control.py` | archived |
| `alamode_sto/STO_RTA_scph_*K.*`, `RTA_scph_*K.*`, `STO_RTA_production.*`, `RTA_production.*`, `PROVENANCE.md` | 4 M | RTA linewidths of the example set (submitted model); inputs of the legacy assembly used by the record tables | ALAMODE anphon RTA | archived |
| `gruneisen_modes/SrTiO3/`, `gruneisen_modes/BaTiO3/` | 332 M | QE matdyn frequencies/eigenvectors of the unstrained and strained cells (`*.modes`, meshes 7–13, on-grid 4³), staging q2r files; BaTiO₃ soft-sector inputs | `scripts/compute_gruneisen.py`, `scripts/gen_matdyn_mesh_inputs.py` | archived |
| `alamode_sto/own_surface_i2s2_noasr_superseded/`, `own_surface_i4s4_noasr_superseded/` | 60 M | diagnostic surface without the distributed sum rule | — | **excluded** (superseded) |
| `alamode_sto/own_surface_i4s4/`, `own_surface_i4s8/` | 24 M | diagnostic surface, SELF_OFFDIAG = 0 layout | — | **excluded** (superseded by `own_od1/`) |
| `alamode_sto/renorm_mesh11_300K_*_renorm.npz` | 11 M | intermediate renormalised-frequency caches of the submitted model | — | **excluded** (superseded) |
| `stale_pbe_fc/` | 1 M | PBE force constants of an earlier functional choice | — | **excluded** (superseded) |
| `ld1_Ba_pbesol/` | 3 M | ld1.x generation of the Ba pseudopotential (contains a UPF file) | QE ld1.x | **excluded** (pseudopotential; source in `dft/qe/pseudopotentials/SOURCES.md`) |

The ALAMODE work directory (`DYNMAT_SCRATCH`, holding the ALAMODE build and the unconverted input sets) is not archived; see
`scripts/scph/README.md` for how to recreate it.
