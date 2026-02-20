"""Lazy-loading HDF5 dataset for the gust common data format.

Reads batches on-the-fly from HDF5 files, yielding (B, C, H, W) jnp arrays.
Multiple fields are stacked into channels.
"""

import threading
import queue

import h5py
import numpy as np
import jax
import jax.numpy as jnp


class PrefetchIterator:
    """Wraps an iterator to prefetch batches in a background thread."""

    def __init__(self, iterator, prefetch_count=2, sharding=None):
        self.iterator = iterator
        self.prefetch_count = prefetch_count
        self.sharding = sharding
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                if self.sharding is not None:
                    item = jax.device_put(item, self.sharding)
                else:
                    item = jax.device_put(item)
                self.queue.put(item)
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __del__(self):
        self.stop_event.set()


class HDF5Dataset:
    """Lazy-loading dataset for the gust common HDF5 format.

    Opens HDF5 file, reads batches on-the-fly, yields (B, C, H, W) jnp arrays.
    Multiple fields are stacked into the channel dimension.

    Args:
        data_path: path to .h5 file in gust common format
        fields: list of field names to load (stacked as channels). If None, uses all fields.
        normalize: if True, compute per-channel mean/std from a subsample and normalize
        batch_size: number of samples per batch
        shuffle: whether to shuffle sample order each epoch
        seed: random seed for shuffling
        prefetch: number of batches to prefetch (0 to disable)
    """

    def __init__(self, data_path, fields=None, normalize=False,
                 batch_size=16, shuffle=True, seed=42, prefetch=2,
                 sharding=None, drop_last=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = prefetch
        self.sharding = sharding
        self.drop_last = drop_last
        self._rng = np.random.RandomState(seed)

        # Open file to read metadata
        with h5py.File(data_path, 'r') as f:
            available = list(f['fields'].keys())
            if fields is None:
                fields = available
            self.fields = fields

            # Validate and get shapes
            shapes = {}
            for name in self.fields:
                if name not in f['fields']:
                    raise ValueError(f"Field '{name}' not found. Available: {available}")
                shapes[name] = f['fields'][name].shape
            self._field_shapes = shapes

            # All fields must have same n_samples and spatial dims
            first = shapes[self.fields[0]]
            self._n_samples = first[0]
            self._spatial_shape = first[1:3]  # (H, W)
            for name, shape in shapes.items():
                if shape[0] != self._n_samples:
                    raise ValueError(f"Field '{name}' has {shape[0]} samples, expected {self._n_samples}")
                if shape[1:3] != self._spatial_shape:
                    raise ValueError(f"Field '{name}' spatial shape {shape[1:3]} != {self._spatial_shape}")

        # Count channels
        self._n_channels = 0
        for name in self.fields:
            shape = self._field_shapes[name]
            if len(shape) == 3:
                self._n_channels += 1  # scalar field
            elif len(shape) == 4:
                self._n_channels += shape[3]  # vector field with C components
            else:
                raise ValueError(f"Field '{name}' has unexpected ndim={len(shape)}")

        # Normalization
        self.mean = None
        self.std = None
        if normalize:
            self._compute_norm_stats()

    def _compute_norm_stats(self, max_samples=1000):
        """Compute per-channel mean/std from a subsample."""
        n = min(max_samples, self._n_samples)
        indices = np.linspace(0, self._n_samples - 1, n, dtype=int)

        channel_data = [[] for _ in range(self._n_channels)]
        with h5py.File(self.data_path, 'r') as f:
            for name in self.fields:
                ds = f['fields'][name]
                for idx in indices:
                    sample = ds[int(idx)]
                    if sample.ndim == 2:
                        channel_data[0].append(sample.flatten())
                    # This simple approach works but let's do it properly
                    # by tracking channel offset

        # Redo properly with channel offset tracking
        channel_sums = np.zeros(self._n_channels, dtype=np.float64)
        channel_sq_sums = np.zeros(self._n_channels, dtype=np.float64)
        channel_counts = np.zeros(self._n_channels, dtype=np.int64)

        with h5py.File(self.data_path, 'r') as f:
            for idx in np.linspace(0, self._n_samples - 1, n, dtype=int):
                c_offset = 0
                for name in self.fields:
                    ds = f['fields'][name]
                    sample = ds[int(idx)].astype(np.float64)
                    if sample.ndim == 2:
                        channel_sums[c_offset] += sample.sum()
                        channel_sq_sums[c_offset] += (sample ** 2).sum()
                        channel_counts[c_offset] += sample.size
                        c_offset += 1
                    elif sample.ndim == 3:
                        for c in range(sample.shape[2]):
                            s = sample[:, :, c]
                            channel_sums[c_offset] += s.sum()
                            channel_sq_sums[c_offset] += (s ** 2).sum()
                            channel_counts[c_offset] += s.size
                            c_offset += 1

        self.mean = (channel_sums / channel_counts).astype(np.float32)
        variance = (channel_sq_sums / channel_counts) - self.mean.astype(np.float64) ** 2
        self.std = np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)
        print(f"Normalization stats: mean={self.mean}, std={self.std}")

    def __len__(self):
        """Number of batches per epoch."""
        if self.drop_last:
            return self._n_samples // self.batch_size
        return (self._n_samples + self.batch_size - 1) // self.batch_size

    def _read_batch(self, f, indices):
        """Read a batch of samples from an open HDF5 file handle.

        Returns (B, C, H, W) float32 numpy array.
        """
        B = len(indices)
        H, W = self._spatial_shape
        batch = np.empty((B, self._n_channels, H, W), dtype=np.float32)

        for b, idx in enumerate(indices):
            c_offset = 0
            for name in self.fields:
                ds = f['fields'][name]
                sample = ds[int(idx)].astype(np.float32)
                if sample.ndim == 2:
                    batch[b, c_offset] = sample
                    c_offset += 1
                elif sample.ndim == 3:
                    for c in range(sample.shape[2]):
                        batch[b, c_offset] = sample[:, :, c]
                        c_offset += 1

        if self.mean is not None:
            batch = (batch - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        return batch

    def _batch_generator(self):
        """Yields (B, C, H, W) numpy arrays."""
        indices = np.arange(self._n_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        with h5py.File(self.data_path, 'r') as f:
            for start in range(0, self._n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if self.drop_last and len(batch_idx) < self.batch_size:
                    break
                # Sort indices for sequential HDF5 access (much faster)
                sorted_order = np.argsort(batch_idx)
                sorted_idx = batch_idx[sorted_order]
                batch = self._read_batch(f, sorted_idx)
                # Unsort to restore shuffled order
                unsort_order = np.argsort(sorted_order)
                batch = batch[unsort_order]
                yield jnp.array(batch)

    def __iter__(self):
        """Yields (B, C, H, W) jnp arrays."""
        gen = self._batch_generator()
        if self.prefetch > 0:
            return PrefetchIterator(gen, prefetch_count=self.prefetch, sharding=self.sharding)
        return gen

    @property
    def sample_shape(self):
        """Returns (C, H, W) for a single sample."""
        H, W = self._spatial_shape
        return (self._n_channels, H, W)

    @property
    def n_samples(self):
        """Total number of samples in the dataset."""
        return self._n_samples
