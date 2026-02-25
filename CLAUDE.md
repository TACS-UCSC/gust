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

## Tokenizer (`models/tokenizer.py`)

Wraps a trained multi-scale VQ-VAE to prepare data for autoregressive model training.

```bash
# Show tokenizer stats (no file written)
python -m models.tokenizer info --checkpoint path/to/model.eqx --config_path path/to/config.txt --data_path data.h5

# Tokenize and save to .npz
python -m models.tokenizer save --checkpoint path/to/model.eqx --config_path path/to/config.txt --data_path data.h5 --output tokens.npz
```

- Two-pass pipeline: `fit()` collects unique codebook indices per scale, then `save` encodes and writes
- Builds a unified per-scale codebook with contiguous index ranges (sparse original indices remapped to consecutive)
- Auto-detects deterministic scales (scales with only 1 unique code) and marks the first trainable scale
- Supports streaming via `create_tokenized_dataloader()` or file-based via `save_tokenized_data()` / `load_tokenized_data()`
- Output `.npz` contains `indices_flat`, unified codebook, per-scale mappings, and the original config

## Reconstruction analysis (`models/analyze_reconstruction.py`)

Evaluates VQ-VAE reconstruction quality via spectral and histogram comparisons against ground truth.

```bash
python -m models.analyze_reconstruction \
  --checkpoint path/to/model.eqx --config_path path/to/config.txt \
  --data_path data.h5 --output_dir ./analysis_output
```

- Computes sample-averaged TKE spectrum E(k) and enstrophy spectrum Z(k), plus pixel value histograms
- Outputs loglog spectral plots (GT vs Reconstruction), histogram plot, and `metrics.json` with MSE, JS divergence, and TV distance
- Supports `--sample_start`/`--sample_stop` for train/test splits and all model architecture overrides

## Skills

Always load the `jax` and `equinox` skills when working in this project.
