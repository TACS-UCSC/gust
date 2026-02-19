"""Common HDF5 read/write utilities for the gust data format.

HDF5 layout:
    /fields/{name}    - (n_samples, H, W) float32 [scalar] or (n_samples, H, W, C) [vector]
    /coordinates/x    - 1D float32
    /coordinates/y    - 1D float32
    /metadata         - group with solver params as attributes
"""

import h5py
import numpy as np


def write_hdf5(output_path, fields_dict, coordinates=None, metadata=None):
    """Write simulation data to common HDF5 format.

    Args:
        output_path: path to .h5 file
        fields_dict: {'field_name': np.ndarray (n_samples, H, W) or (n_samples, H, W, C)}
        coordinates: optional {'x': 1D array, 'y': 1D array}
        metadata: optional dict stored as HDF5 attributes on /metadata group
    """
    with h5py.File(output_path, 'w') as f:
        fields_grp = f.create_group('fields')
        for name, data in fields_dict.items():
            data = np.asarray(data, dtype=np.float32)
            fields_grp.create_dataset(name, data=data, chunks=True, compression='gzip')

        if coordinates is not None:
            coords_grp = f.create_group('coordinates')
            for name, data in coordinates.items():
                coords_grp.create_dataset(name, data=np.asarray(data, dtype=np.float32))

        meta_grp = f.create_group('metadata')
        if metadata is not None:
            for key, val in metadata.items():
                try:
                    meta_grp.attrs[key] = val
                except TypeError:
                    meta_grp.attrs[key] = str(val)


def read_hdf5_info(path):
    """Return dict of field names, shapes, and metadata without loading data.

    Returns:
        dict with keys:
            'fields': {name: {'shape': tuple, 'dtype': str}}
            'coordinates': {name: {'shape': tuple}} or {}
            'metadata': dict of attributes
    """
    info = {'fields': {}, 'coordinates': {}, 'metadata': {}}
    with h5py.File(path, 'r') as f:
        if 'fields' in f:
            for name in f['fields']:
                ds = f['fields'][name]
                info['fields'][name] = {'shape': ds.shape, 'dtype': str(ds.dtype)}
        if 'coordinates' in f:
            for name in f['coordinates']:
                ds = f['coordinates'][name]
                info['coordinates'][name] = {'shape': ds.shape}
        if 'metadata' in f:
            for key, val in f['metadata'].attrs.items():
                info['metadata'][key] = val
    return info


def read_hdf5_field(path, field, start=None, stop=None):
    """Read a slice of a field. Returns numpy array.

    Args:
        path: path to .h5 file
        field: field name under /fields/
        start: starting sample index (inclusive), None for beginning
        stop: ending sample index (exclusive), None for end

    Returns:
        np.ndarray of shape (n, H, W) or (n, H, W, C)
    """
    with h5py.File(path, 'r') as f:
        ds = f['fields'][field]
        if start is None and stop is None:
            return ds[:]
        s = slice(start, stop)
        return ds[s]
