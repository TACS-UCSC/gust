"""CLI wrapper to run Rayleigh-Benard convection simulations.

Runs the Dedalus solver and then auto-converts to the gust common HDF5 format
(output.h5 in --output_dir).

Example:
    python run_solver.py --output_dir ./rbc_output --rayleigh 1e6 --prandtl 1.0 --dT 1.0
    python run_solver.py --output_dir ./rbc_output --resolution 512x128 --seed 42
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from ..common.hdf5_utils import write_hdf5


def find_dedalus_h5_files(dedalus_dir):
    """Find and sort Dedalus snapshot HDF5 files in a directory."""
    dedalus_dir = Path(dedalus_dir)
    h5_files = sorted(dedalus_dir.glob('*.h5'))
    if not h5_files:
        # Dedalus sometimes puts files in a subdirectory
        for subdir in dedalus_dir.iterdir():
            if subdir.is_dir():
                h5_files = sorted(subdir.glob('*.h5'))
                if h5_files:
                    break
    return h5_files


def convert_dedalus_to_hdf5(dedalus_dir, output_path,
                            scalar_fields=('buoyancy', 'vorticity'),
                            velocity=False):
    """Convert Dedalus snapshot output to gust common HDF5 format."""
    h5_files = find_dedalus_h5_files(dedalus_dir)
    if not h5_files:
        print(f"No .h5 files found in {dedalus_dir}, skipping conversion.")
        return

    print(f"Converting {len(h5_files)} Dedalus snapshot files to common HDF5...")

    all_data = {name: [] for name in scalar_fields}
    if velocity:
        all_data['velocity_x'] = []
        all_data['velocity_z'] = []

    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            if 'tasks' not in f:
                continue
            tasks = f['tasks']
            for field_name in scalar_fields:
                if field_name in tasks:
                    data = np.array(tasks[field_name], dtype=np.float32)
                    if data.ndim == 3:
                        all_data[field_name].append(data)
            if velocity and 'velocity' in tasks:
                vel = np.array(tasks['velocity'], dtype=np.float32)
                if vel.ndim == 4:
                    all_data['velocity_x'].append(vel[:, 0, :, :])
                    all_data['velocity_z'].append(vel[:, 1, :, :])

    fields_dict = {}
    for name, chunks in all_data.items():
        if chunks:
            stacked = np.concatenate(chunks, axis=0)
            fields_dict[name] = stacked
            print(f"  {name}: {stacked.shape}")

    if not fields_dict:
        print("No data extracted from Dedalus output, skipping conversion.")
        return

    # Get coordinates from the first file
    coordinates = {}
    with h5py.File(h5_files[0], 'r') as f:
        if 'scales' in f:
            for key in f['scales']:
                if 'x' in key.lower():
                    coordinates['x'] = np.array(f['scales'][key], dtype=np.float32)
                elif 'z' in key.lower():
                    coordinates['y'] = np.array(f['scales'][key], dtype=np.float32)

    n_samples = list(fields_dict.values())[0].shape[0]
    metadata = {
        'solver': 'rayleigh_benard',
        'fields': ','.join(fields_dict.keys()),
        'n_samples': n_samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_hdf5(output_path, fields_dict, coordinates=coordinates, metadata=metadata)
    print(f"Wrote {len(fields_dict)} fields, {n_samples} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Rayleigh-Benard convection simulation via Dedalus')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save HDF5 output')
    parser.add_argument('--rayleigh', type=float, default=1e6,
                        help='Rayleigh number')
    parser.add_argument('--prandtl', type=float, default=1.0,
                        help='Prandtl number')
    parser.add_argument('--dT', type=float, default=1.0,
                        help='Temperature difference for linear background')
    parser.add_argument('--resolution', type=str, default='512x128',
                        help='Resolution as NXxNZ (e.g., 512x128)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initial condition noise')
    parser.add_argument('--init', type=str, default='default',
                        help='Initial condition type')
    parser.add_argument('--safety_factor', type=int, default=32,
                        help='CFL safety divisor (higher = smaller dt)')
    parser.add_argument('--stop_sim_time', type=float, default=50,
                        help='Total simulation time')
    parser.add_argument('--snapshot_dt', type=float, default=0.25,
                        help='Time between saved snapshots')
    parser.add_argument('--max_writes', type=int, default=200,
                        help='Max snapshots per Dedalus HDF5 file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Parse resolution
    parts = args.resolution.lower().split('x')
    if len(parts) != 2:
        raise ValueError(f"Resolution must be NXxNZ, got: {args.resolution}")
    resolution = (int(parts[0]), int(parts[1]))

    dpath = Path(args.output_dir)
    dpath.mkdir(parents=True, exist_ok=True)

    from .generate_rbc import generate_rayleigh_benard

    print(f"Running RBC: Ra={args.rayleigh:.1e}, Pr={args.prandtl}, dT={args.dT}, "
          f"resolution={resolution}, seed={args.seed}")

    generate_rayleigh_benard(
        resolution=resolution,
        rayleigh=args.rayleigh,
        prandtl=args.prandtl,
        init=args.init,
        seed=args.seed,
        dT=args.dT,
        dpath=dpath,
        safety_factor=args.safety_factor,
        min_dt=1e-8,
        stop_sim_time=args.stop_sim_time,
        snapshot_dt=args.snapshot_dt,
        max_writes=args.max_writes,
    )

    print(f"Dedalus simulation complete. Converting to common HDF5...")

    # Auto-convert Dedalus output to common HDF5
    h5_output = dpath / 'output.h5'
    convert_dedalus_to_hdf5(dpath, str(h5_output))

    print(f"Done. Output in: {h5_output}")


if __name__ == '__main__':
    main()
