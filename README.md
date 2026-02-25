# gust

Solvers, dataloaders, and tokenization pipeline for 2D fluid dynamics scaling experiments. Generates training data for VQ-VAE and autoregressive models.

## Overview

Two physics solvers produce 2D field data (turbulence, convection) directly in a common HDF5 format, a lazy-loading dataloader feeds batches to JAX models, and a tokenizer wraps a trained multi-scale VQ-VAE to produce discrete token sequences for autoregressive model training.

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
  models/
    models.py                          # Multi-scale VQ-VAE (VQVAE2d)
    train_vqvae.py                     # VQ-VAE training script
    tokenizer.py                       # VQ-VAE tokenizer for AR data prep
    analyze_reconstruction.py          # Spectral & histogram reconstruction analysis
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

## Tokenizer

The tokenizer (`models/tokenizer.py`) wraps a trained multi-scale VQ-VAE checkpoint to encode field data into discrete token sequences for autoregressive model training.

```bash
# Show tokenizer stats and verify round-trip encoding
python -m models.tokenizer info \
  --checkpoint path/to/model.eqx --config_path path/to/config.txt \
  --data_path data.h5

# Tokenize dataset and save to .npz
python -m models.tokenizer save \
  --checkpoint path/to/model.eqx --config_path path/to/config.txt \
  --data_path data.h5 --output tokens.npz
```

The tokenizer builds a unified per-scale codebook where each scale gets a contiguous index range, remapping sparse codebook usage to consecutive indices. The output `.npz` file contains flattened indices, the unified codebook, per-scale mappings, and the original model config.

```python
from models.tokenizer import load_tokenized_data

data = load_tokenized_data("tokens.npz")
# data["indices_flat"]          — (N, total_tokens) discrete targets
# data["vectors_flat"]          — (N, total_tokens, D) codebook vectors
# data["codebook"]              — (effective_vocab, D) unified codebook
# data["effective_vocab_size"]  — int
```

## Reconstruction Analysis

Evaluate VQ-VAE reconstruction quality with spectral and statistical comparisons:

```bash
python -m models.analyze_reconstruction \
  --checkpoint path/to/model.eqx --config_path path/to/config.txt \
  --data_path data.h5 --output_dir ./analysis_output \
  --sample_start 0 --sample_stop 10000
```

Produces `tke_spectrum.png`, `enstrophy_spectrum.png`, `pixel_histogram.png`, and `metrics.json` (MSE, JS divergence, TV distance) in the output directory.

## Solvers

| Solver | Domain | Resolution | Fields | ~Timesteps |
|--------|--------|-----------|--------|-----------|
| Py2D Turbulence | Square periodic | 512x512 | vorticity | 125K |
| Rayleigh-Benard | Rectangular (periodic x walls) | 512x128 | buoyancy, vorticity | 200/file |

See [solvers/README.md](solvers/README.md) for per-solver usage.
