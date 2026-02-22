# CLAUDE.md

## Project

Gust: solvers + dataloaders for 2D fluid dynamics data generation. Two solvers (py2d_turbulence, rayleigh_benard) produce simulation data directly in common HDF5 format, and dataloaders feed JAX models.

## Running solvers

All solvers run from the repo root as modules and output HDF5 directly:

```bash
python -m solvers.py2d_turbulence.run_solver --output_dir ./output --Re 1000 --NX 256
python -m solvers.rayleigh_benard.run_solver --output_dir ./output --rayleigh 1e6
```

## Dependencies

Single `requirements.txt` at repo root. Install with `pip install -r requirements.txt`.

## Imports

- Use relative imports within solver packages (e.g., `from ..common.hdf5_utils import write_hdf5`)
- No `sys.path` hacks

## Key conventions

- Solver output goes to user-specified `--output_dir` (no hardcoded paths)
- py2d_turbulence uses vendored py2d routines in `py2d_core.py` (MIT, from github.com/envfluids/py2d)
- Common HDF5 format: `/fields/{name}` as `(n_samples, H, W) float32`
- JAX for array computing

## VQ-VAE training (`models/`)

- Gradient clipping: `optax.clip_by_global_norm(0.5)` is applied before AdamW in the optimizer chain (`train_vqvae.py`)
- L2 normalization: encoder outputs and codebook vectors are L2-normalized before the distance computation in `_quantize_single` (`models.py`). This bounds commitment loss and prevents magnitude-driven spikes.

## Skills

Always load the `jax` and `equinox` skills when working in this project.
