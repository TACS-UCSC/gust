# gust

Solvers and dataloaders for 2D fluid dynamics scaling experiments. Generates training data for VQ-VAE and autoregressive models.

## Overview

Two physics solvers produce 2D field data (turbulence, convection) directly in a common HDF5 format, and a lazy-loading dataloader feeds batches to JAX models.

## Directory Structure

```
gust/
  requirements.txt                     # All dependencies
  solvers/
    common/
      hdf5_utils.py                    # Common HDF5 read/write utilities
    py2d_turbulence/
      run_solver.py                    # 2D forced turbulence (vendored py2d)
    rayleigh_benard/
      run_solver.py                    # Rayleigh-Benard convection (Dedalus)
      generate_rbc.py                  # Core Dedalus solver
    README.md                          # Per-solver docs
  dataloaders/
    hdf5_dataset.py                    # Lazy-loading HDF5 dataset + prefetch
```

## Quickstart

```bash
# Install all dependencies
pip install -r requirements.txt

# Run a solver (from repo root) — each produces output.h5 directly
python -m solvers.py2d_turbulence.run_solver --output_dir ./py2d_output --Re 1000 --NX 256 --tTotal 5.0
python -m solvers.rayleigh_benard.run_solver --output_dir ./rbc_output --rayleigh 1e6 --resolution 512x128
```

## Common HDF5 Format

All solvers produce data in a shared format:

```
/fields/{name}    - (n_samples, H, W) float32
/coordinates/x    - 1D float32
/coordinates/y    - 1D float32
/metadata         - solver params as attributes
```

## Dataloader

```python
from dataloaders import HDF5Dataset

# Lazy-load from HDF5, yields (B, C, H, W) jnp arrays
dataset = HDF5Dataset('data.h5', batch_size=16, normalize=True)

print(f"Samples: {dataset.n_samples}, Shape: {dataset.sample_shape}")

for batch in dataset:
    print(batch.shape)  # (16, C, H, W)
    break
```

Multiple fields are stacked as channels. Normalization computes per-channel mean/std from a subsample.

## Solvers

| Solver | Domain | Resolution | Fields | ~Timesteps |
|--------|--------|-----------|--------|-----------|
| Py2D Turbulence | Square periodic | 512x512 | vorticity | 125K |
| Rayleigh-Benard | Rectangular (periodic x walls) | 512x128 | buoyancy, vorticity | 200/file |

See [solvers/README.md](solvers/README.md) for per-solver usage.
