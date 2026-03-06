"""Generate MP4 video from rollout fields.

Renders side-by-side panels: Raw GT | Tokenized GT | Prediction | Difference.

Usage:
    python -m models.video_rollout \
        --fields_path rollout_1000/rollout_fields.npz \
        --data_path output/output.h5 \
        --start_frame 0 \
        --fps 50 \
        --output_dir rollout_1000
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import imageio.v3 as iio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rollout video")

    parser.add_argument("--fields_path", type=str, default=None,
                        help="Path to rollout_fields.npz (pre-decoded)")
    parser.add_argument("--tokens_path", type=str, default=None,
                        help="Path to rollout_tokens.npz (tokenizer-compatible)")
    parser.add_argument("--gt_tokens_path", type=str, default=None,
                        help="Path to gt_tokens.npz (tokenizer-compatible)")

    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to HDF5 data file for raw GT fields")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Start frame index (auto-detected from tokens if available)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Only use the first N frames")

    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    parser.add_argument("--vqvae_config", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="./rollout_output")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video frames per second")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Video resolution (DPI)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_raw_fields(data_path, field, start_frame, n_frames):
    """Load raw (pre-tokenization) fields from HDF5."""
    import h5py
    with h5py.File(data_path, "r") as f:
        ds = f["fields"][field]
        end = min(start_frame + n_frames, ds.shape[0])
        data = ds[start_frame:end].astype(np.float32)
    return data


def decode_tokens_to_fields(tokens_npz_path, vqvae_checkpoint, vqvae_config_path):
    """Decode a tokenizer-compatible .npz to fields via VQ-VAE."""
    from .tokenizer import (
        load_tokenized_data, load_config, load_vqvae_checkpoint,
        unflatten_to_scales,
    )

    token_data = load_tokenized_data(tokens_npz_path)
    indices = token_data["indices_flat"]
    scales = token_data["scales"]
    unified_to_original = jnp.array(token_data["unified_to_original"])

    config = load_config(vqvae_config_path)
    key = jax.random.PRNGKey(42)
    vqvae = load_vqvae_checkpoint(vqvae_checkpoint, config, key)
    vqvae = eqx.nn.inference_mode(vqvae)

    fields = []
    for i in range(len(indices)):
        flat = jnp.array(indices[i])
        idx_list = unflatten_to_scales(flat, scales)
        orig_idx = [unified_to_original[idx] for idx in idx_list]
        field = vqvae.decode_indices(orig_idx)
        fields.append(np.array(field))
        if (i + 1) % 100 == 0:
            print(f"    Decoded {i + 1}/{len(indices)}")

    return np.stack(fields)


def _fig_to_rgb(fig):
    """Render a matplotlib figure to an [H, W, 3] uint8 numpy array."""
    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()


def make_video(gt_raw, gt_tok, pred, output_path, start_frame=0, fps=20, dpi=120):
    """Render side-by-side video: Raw GT | Tokenized GT | Prediction | Diff."""
    n_frames = len(pred)

    has_raw = gt_raw is not None
    has_tok = gt_tok is not None
    n_panels = 1 + int(has_raw) + int(has_tok) + 1

    def get2d(arr, i):
        return arr[i, 0] if arr.ndim == 4 else arr[i]

    ref = gt_raw if has_raw else gt_tok
    vmin, vmax = float(ref.min()), float(ref.max())

    pred_2d_all = pred[:, 0] if pred.ndim == 4 else pred
    ref_2d_all = ref[:, 0] if ref.ndim == 4 else ref
    d_abs = max(float(np.abs(pred_2d_all - ref_2d_all).max()), 1e-6)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), dpi=dpi)
    fig.canvas = FigureCanvasAgg(fig)
    if n_panels == 1:
        axes = [axes]

    imgs = []
    col = 0
    if has_raw:
        im = axes[col].imshow(ref[0] if ref.ndim == 3 else ref[0, 0],
                               cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[col].set_title("Ground Truth (raw)", fontsize=10)
        axes[col].axis("off")
        imgs.append(("raw", im))
        col += 1
    if has_tok:
        im = axes[col].imshow(get2d(gt_tok, 0), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[col].set_title("GT (tokenized)", fontsize=10)
        axes[col].axis("off")
        imgs.append(("tok", im))
        col += 1

    im = axes[col].imshow(get2d(pred, 0), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[col].set_title("Prediction", fontsize=10)
    axes[col].axis("off")
    imgs.append(("pred", im))
    col += 1

    im = axes[col].imshow(get2d(pred, 0) - (ref[0] if ref.ndim == 3 else ref[0, 0]),
                            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)
    axes[col].set_title("Difference", fontsize=10)
    axes[col].axis("off")
    imgs.append(("diff", im))

    suptitle = fig.suptitle(f"Frame {start_frame}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    t0 = time.time()
    with iio.imopen(output_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)

        for i in range(n_frames):
            ref_i = ref[i] if ref.ndim == 3 else get2d(ref, i)

            for kind, im_obj in imgs:
                if kind == "raw":
                    im_obj.set_data(gt_raw[i] if gt_raw.ndim == 3 else get2d(gt_raw, i))
                elif kind == "tok":
                    im_obj.set_data(get2d(gt_tok, i))
                elif kind == "pred":
                    im_obj.set_data(get2d(pred, i))
                elif kind == "diff":
                    im_obj.set_data(get2d(pred, i) - ref_i)

            suptitle.set_text(f"Frame {start_frame + i}")
            rgb = _fig_to_rgb(fig)
            writer.write_frame(rgb)

            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"  Rendered {i + 1}/{n_frames} frames "
                      f"({elapsed:.0f}s, {(i+1)/elapsed:.1f} fps)")

    plt.close(fig)
    print(f"  Saved video to {output_path}")


def _load_fields(args):
    """Load rollout and GT tokenized fields."""
    if args.fields_path:
        print(f"Loading decoded fields from {args.fields_path}...")
        data = np.load(args.fields_path)
        rollout_fields = data["rollout"]
        gt_tok_fields = data["ground_truth"]
        n_frames = len(rollout_fields)
        print(f"  {n_frames} frames, shape {rollout_fields.shape}")
        return rollout_fields, gt_tok_fields, n_frames

    if args.tokens_path:
        if not (args.vqvae_checkpoint and args.vqvae_config):
            raise ValueError("--vqvae_checkpoint and --vqvae_config required "
                             "when using --tokens_path")
        print(f"Decoding rollout tokens from {args.tokens_path}...")
        rollout_fields = decode_tokens_to_fields(
            args.tokens_path, args.vqvae_checkpoint, args.vqvae_config)
        n_frames = len(rollout_fields)

        gt_tok_fields = None
        if args.gt_tokens_path:
            print(f"Decoding GT tokens from {args.gt_tokens_path}...")
            gt_tok_fields = decode_tokens_to_fields(
                args.gt_tokens_path, args.vqvae_checkpoint, args.vqvae_config)
        return rollout_fields, gt_tok_fields, n_frames

    raise ValueError("Provide either --fields_path or --tokens_path")


def _detect_start_frame(args):
    """Auto-detect start_frame from npz metadata."""
    if args.start_frame is not None:
        return args.start_frame
    for path in [args.tokens_path, args.fields_path]:
        if path is None:
            continue
        try:
            d = np.load(path, allow_pickle=True)
            if "rollout_start_frame" in d:
                return int(d["rollout_start_frame"])
            if "start_frame" in d:
                return int(d["start_frame"])
        except Exception:
            pass
    print("  Warning: could not detect start_frame, defaulting to 0")
    return 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rollout_fields, gt_tok_fields, n_frames = _load_fields(args)
    start_frame = _detect_start_frame(args)
    print(f"  Start frame: {start_frame}, n_frames: {n_frames}")

    if args.max_frames is not None and args.max_frames < n_frames:
        n_frames = args.max_frames
        rollout_fields = rollout_fields[:n_frames]
        if gt_tok_fields is not None:
            gt_tok_fields = gt_tok_fields[:n_frames]
        print(f"  Truncated to {n_frames} frames (--max_frames)")

    gt_raw_fields = None
    if args.data_path:
        print(f"Loading raw GT fields from {args.data_path}...")
        gt_raw_fields = load_raw_fields(
            args.data_path, args.field, start_frame, n_frames)
        print(f"  Loaded {len(gt_raw_fields)} frames, shape {gt_raw_fields.shape}")
        n_frames = min(n_frames, len(gt_raw_fields))
        gt_raw_fields = gt_raw_fields[:n_frames]
        rollout_fields = rollout_fields[:n_frames]
        if gt_tok_fields is not None:
            gt_tok_fields = gt_tok_fields[:n_frames]

    print("\nGenerating video...")
    video_path = os.path.join(args.output_dir, "rollout.mp4")
    make_video(
        gt_raw=gt_raw_fields,
        gt_tok=gt_tok_fields,
        pred=rollout_fields,
        output_path=video_path,
        start_frame=start_frame,
        fps=args.fps,
        dpi=args.dpi,
    )

    print("\nVideo generation complete.")


if __name__ == "__main__":
    main()
