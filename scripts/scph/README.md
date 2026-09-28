# scripts/scph — SCPH surfaces of SrTiO₃

These scripts build the harmonic sets of the hybrid model (production) and of the diagnostic surface, and drive ALAMODE
anphon (SCPH, dfc2, RTA) for the unstrained and shear-strained sets. The outputs land in `data/raw/alamode_sto/`
(gitignored; archived on Zenodo, see `data/raw/MANIFEST.md`). All commands run from the repository root.

| file | role |
|---|---|
| `build_hybrid_model.py` | `reexpress`: example harmonic set on the 4×4×4 supercell; `strain`: strained sets D_tut ± h K_QE; `qefc`: q2r-format copies for the assembly code |
| `q2r_to_alamode_fc2.py` | QE q2r force constants → ALAMODE FC2 XML (+ BORN), dipole term restored, distributed sum rule |
| `run_scph_surface.py` | anphon SCPH (SELF_OFFDIAG = 1) per set and mesh, dfc2 renormalised FC2, 11³ eigenvectors, 8³ RTA of the unstrained set; stops and renames outputs `*.unconverged` if SCPH does not converge |
| `anphon_io.py` | anphon input writer and log/eigenvector parsers |

## 1. Work directory and ALAMODE 1.5.0

The long runs use a work directory given by `DYNMAT_SCRATCH` (not archived). macOS with Homebrew (`eigen`, `libomp`, `fftw`,
`spglib`, `boost`, an MPI):

```bash
export DYNMAT_SCRATCH=$HOME/alamode_work          # any location
cd "$DYNMAT_SCRATCH"
git clone https://github.com/ttadano/alamode.git && git -C alamode checkout v.1.5.0   # d86ab0f1
cmake -S alamode -B alamode/_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS="-I$(brew --prefix eigen)/include/eigen3 -I$(brew --prefix libomp)/include -Xpreprocessor -fopenmp" \
  -DCMAKE_C_FLAGS="-I$(brew --prefix libomp)/include -Xpreprocessor -fopenmp" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib -lomp -L$(brew --prefix fftw)/lib" \
  -DSPGLIB_ROOT=$(brew --prefix spglib)
cmake --build alamode/_build -j10
export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$(brew --prefix fftw)/lib
# the example quartic set used by every run (checksums in data/processed/v3_diagnostics/refit_provenance.md)
bunzip2 -k alamode/example/SrTiO3/reference/STO_anharm.xml.bz2
mkdir -p run/own_surface_asr
ln -s "$DYNMAT_SCRATCH/alamode/example/SrTiO3/reference/STO_anharm.xml" run/own_surface_asr/STO_anharm.xml
cd -    # back to the repository root
```

## 2. Hybrid model (production)

```bash
uv run python scripts/scph/build_hybrid_model.py reexpress   # -> $DYNMAT_SCRATCH/run/z_sets/STO444_tut.xml, BORN_tut
uv run python scripts/scph/build_hybrid_model.py strain      # -> z_{reference,shear_xy_[pm]0{05,10}}_full_fc2.xml, BORN_z_*
uv run python scripts/scph/build_hybrid_model.py qefc        # -> data/raw/alamode_sto/z_tut/z_*.fc
```

SCPH at 300 K. Unstrained set with MIXALPHA 0.2 (default), strained sets with 0.1; retry a non-converged set with 0.05. The
quartic vertex array is replicated on every MPI rank, so the inner mesh sets the rank count (12³: one rank; 36 GB machine):

```bash
Z="--alat 7.363 --xml-dir $DYNMAT_SCRATCH/run/z_sets --maxiter 400"
for M in i2s2:4:2 i2s4:4:2 i2s8:2:5 i2s12:1:10; do
  mesh=${M%%:*}; r=${M#*:}; np=${r%%:*}; omp=${r#*:}
  uv run python scripts/scph/run_scph_surface.py --mesh $mesh $Z --np $np --omp $omp \
      --sets z_reference --out data/raw/alamode_sto/z_tut/$mesh
  uv run python scripts/scph/run_scph_surface.py --mesh $mesh $Z --np $np --omp $omp --mixalpha 0.1 --skip-rta \
      --sets z_shear_xy_p005 z_shear_xy_m005 z_shear_xy_p010 z_shear_xy_m010 --out data/raw/alamode_sto/z_tut/$mesh
done
# 2^3/16^3: unstrained set only (the strained 16^3 runs need about 40 GB peak)
uv run python scripts/scph/run_scph_surface.py --mesh i2s16 $Z --np 1 --omp 10 --sets z_reference \
    --out data/raw/alamode_sto/z_tut/i2s16
# temperature series at 2^3/12^3
for T in 250 350 400; do
  uv run python scripts/scph/run_scph_surface.py --mesh i2s12 $Z --np 1 --omp 10 --mixalpha 0.1 --temps $T \
      --label i2s12T$T --sets z_reference z_shear_xy_p005 z_shear_xy_m005 --out data/raw/alamode_sto/z_tut/i2s12_T
done
# 250 K, z_shear_xy_m005: repeat with --mixalpha 0.05. 200 K: the strained set does not converge (recorded, not used).
```

The harmonic RTA reference (`z_tut/bare/`) is anphon `MODE = RTA` on `STO444_tut.xml` without SCPH, 8³ mesh, 300 K.

Then, in `data/processed/`: `uv run python scripts/compute_eta_SrTiO3.py` (production η_xyxy(T)), followed by the table and
figure scripts listed in the top-level `README.md`.

## 3. Diagnostic surface (own QE-PBEsol harmonic set)

```bash
mkdir -p "$DYNMAT_SCRATCH/run/convert"
for s in reference shear_xy_p005 shear_xy_m005 shear_xy_p010 shear_xy_m010; do
  fc=dft/qe/SrTiO3/gruneisen_pbesol/$s/SrTiO3_pbesol_$s.444.fc
  [ $s = reference ] && fc=dft/qe/SrTiO3/dispersion_pbesol/SrTiO3_pbesol.444.fc
  uv run python scripts/scph/q2r_to_alamode_fc2.py $fc "$DYNMAT_SCRATCH/run/convert/${s}_full_fc2.xml" \
      --asr distributed --add-dipole --born "$DYNMAT_SCRATCH/run/convert/BORN_$s"
done
for M in i2s2:4:2 i2s4:4:2 i2s8:2:5 i2s12:1:10 i3s12:1:10 i4s4:4:2 i4s8:2:5 i4s12:1:10; do
  mesh=${M%%:*}; r=${M#*:}; np=${r%%:*}; omp=${r#*:}
  uv run python scripts/scph/run_scph_surface.py --mesh $mesh --np $np --omp $omp \
      --sets reference --out data/raw/alamode_sto/own_od1/$mesh
  uv run python scripts/scph/run_scph_surface.py --mesh $mesh --np $np --omp $omp --mixalpha 0.1 --skip-rta \
      --sets shear_xy_p005 shear_xy_m005 --out data/raw/alamode_sto/own_od1/$mesh
done
```

The 2³/2³ runs cover 100–400 K (`--temps 100 150 200 250 300 350 400`). The s = ±0.010 pair (h = ±0.020) of the diagnostic surface was run at 2³/2³ only (`--sets shear_xy_p010 shear_xy_m010`); its outputs are in `data/raw/alamode_sto/own_surface_6p2/`. Temperature-stability runs: `--temps 200 250 300
350 400` with outputs in `own_od1/i4s8_Tstab` and `own_od1/i4s12_Tstab`. Symmetrised strained inputs:
`scripts/v3_diagnostics/make_symmetrised_inputs.py`, then `run_scph_surface.py --mesh i2s2 --label i2s2sym --xml-dir <its
output> --out data/raw/alamode_sto/own_od1/i2s2_sym`.

## 4. Example-set controls

`scripts/v3_diagnostics/benchmark_variants_11p0.py` and `scripts/v3_diagnostics/tutorial_kmesh_control.py` run anphon on the
unmodified example set (`STO222.xml` + `STO_anharm.xml`) and write to `data/raw/alamode_sto/example_set_benchmark/`.

## Notes

- Mesh tags: `iNsM` = `KMESH_INTERPOLATE` N³, `KMESH_SCPH` M³. The correction mesh 3³ is not used for results: the
  Γ-centred 3³ grid contains neither R nor M.
- Cell parameters: 7.363 bohr for the hybrid model and the example set, 7.356754 bohr for the diagnostic surface.
- Each run folder gets `runs.csv` with wall time, iterations and the final SCPH residual.
