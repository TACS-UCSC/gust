"""Convert vorticity HDF5 snapshots to MP4 video.

Reads fields/omega from the HDF5 file produced by run_solver and renders
each frame as a colormapped image, writing an MP4 via ffmpeg.

Example:
    python -m solvers.py2d_turbulence.make_video --input ./output/output.h5
    python -m solvers.py2d_turbulence.make_video --input ./output/output.h5 --fps 60 --cmap viridis
"""

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def main():
    parser = argparse.ArgumentParser(
        description="Render vorticity snapshots from HDF5 to MP4 video")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to HDF5 file with fields/omega dataset")
    parser.add_argument("--output", type=str, default="video.mp4",
                        help="Output MP4 path (default: video.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--cmap", type=str, default="RdBu_r",
                        help="Matplotlib colormap (default: RdBu_r)")
    parser.add_argument("--vrange", type=float, default=None,
                        help="Symmetric color limits [-vrange, vrange]. "
                             "If not set, derived from data.")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as h5f:
        omega = h5f["fields/omega"][:]  # (n_samples, H, W)

    n_frames = omega.shape[0]
    print(f"Loaded {n_frames} frames of shape {omega.shape[1:]}")

    if args.vrange is not None:
        vmin, vmax = -args.vrange, args.vrange
    else:
        abs_max = max(abs(float(omega.min())), abs(float(omega.max())))
        vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots()
    img = ax.imshow(omega[0], origin="lower", cmap=args.cmap,
                    vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax, label="vorticity")
    ax.set_title("frame 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame_idx):
        img.set_data(omega[frame_idx])
        ax.set_title(f"frame {frame_idx}")
        return [img]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    writer = FFMpegWriter(fps=args.fps)
    anim.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Saved {args.output} ({n_frames} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
