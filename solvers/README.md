# Solvers

Two physics solvers that generate 2D fluid dynamics data directly in common HDF5 format.

## Common HDF5 Format

All solvers produce files with this layout:

```
/fields/{name}    - (n_samples, H, W) float32    [scalar fields]
                    (n_samples, H, W, C) float32  [vector fields]
/coordinates/x    - 1D float32
/coordinates/y    - 1D float32
/metadata         - group with solver params as attributes
```

## Py2D Turbulence

2D forced turbulence on a square periodic domain. Uses vendored routines from [Py2D-CFD](https://github.com/envfluids/py2d) (MIT license).

```bash
python -m solvers.py2d_turbulence.run_solver \
    --output_dir ./py2d_output \
    --Re 1000 --NX 512 --fkx 4 --fky 4 \
    --dt 1e-4 --tTotal 20.0 --tSave 1e-2
```

Produces `py2d_output/output.h5` with `/fields/omega` of shape `(N, H, W)`.

## Rayleigh-Benard Convection

2D Boussinesq convection using [Dedalus](https://dedalus-project.org/). Fourier (x) + Chebyshev (z), no-slip walls.

```bash
python -m solvers.rayleigh_benard.run_solver \
    --output_dir ./rbc_output \
    --rayleigh 1e6 --prandtl 1.0 --dT 1.0 \
    --resolution 512x128 --seed 42
```

Produces `rbc_output/output.h5` with `/fields/{buoyancy,vorticity}`.

## End-to-End Example

```bash
# 1. Generate data (produces output.h5 directly)
python -m solvers.py2d_turbulence.run_solver --output_dir ./data/py2d --Re 1000 --NX 256 --tTotal 5.0

# 2. Load with dataloader
python -c "
from dataloaders import HDF5Dataset
ds = HDF5Dataset('./data/py2d/output.h5', batch_size=16)
print(f'Samples: {ds.n_samples}, Shape: {ds.sample_shape}')
for batch in ds:
    print(f'Batch: {batch.shape}')
    break
"
```
