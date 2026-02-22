import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import typing as tp


class ResBlock2d(eqx.Module):
    """Residual block with 2D convolutions and optional GroupNorm."""
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    norm1: tp.Optional[nn.GroupNorm]
    norm2: tp.Optional[nn.GroupNorm]
    use_norm: bool = eqx.field(static=True)

    def __init__(self, dim: int, use_norm: bool = False, num_groups: int = 32, key=None):
        key1, key2 = jax.random.split(key, 2)
        self.use_norm = use_norm

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, key=key1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, key=key2)

        if use_norm:
            groups = min(num_groups, dim)
            self.norm1 = nn.GroupNorm(groups=groups, channels=dim)
            self.norm2 = nn.GroupNorm(groups=groups, channels=dim)
        else:
            self.norm1 = None
            self.norm2 = None

    def __call__(self, x):
        y = x
        if self.use_norm:
            y = self.norm1(y)
        y = jax.nn.silu(y)
        y = self.conv1(y)
        if self.use_norm:
            y = self.norm2(y)
        y = jax.nn.silu(y)
        y = self.conv2(y)
        return y + x


class SelfAttention2d(eqx.Module):
    """Self-attention for 2D feature maps."""
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    to_qkv: nn.Conv2d
    to_out: nn.Conv2d
    norm: tp.Optional[nn.GroupNorm]

    def __init__(self, dim: int, num_heads: int = 8, use_norm: bool = True, key=None):
        key1, key2 = jax.random.split(key, 2)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, key=key1)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, key=key2)
        self.norm = nn.GroupNorm(groups=min(32, dim), channels=dim) if use_norm else None

    def __call__(self, x):
        # x: [C, H, W]
        C, H, W = x.shape
        residual = x

        if self.norm is not None:
            x = self.norm(x)

        qkv = self.to_qkv(x)  # [3*C, H, W]
        # Two-step reshape to avoid ShardingTypeError with multi-device
        # (split channel axis first, then merge spatial axes)
        qkv = jnp.reshape(qkv, (3, self.num_heads, self.head_dim, H, W))  # [3, heads, head_dim, H, W]
        qkv = jnp.reshape(qkv, (3, self.num_heads, self.head_dim, H * W))  # [3, heads, head_dim, N]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [heads, head_dim, N]

        # Attention: [heads, N, N]
        q = jnp.transpose(q, (0, 2, 1))  # [heads, N, head_dim]
        k = jnp.transpose(k, (0, 2, 1))  # [heads, N, head_dim]
        v = jnp.transpose(v, (0, 2, 1))  # [heads, N, head_dim]

        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) * self.scale  # [heads, N, N]
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v)  # [heads, N, head_dim]
        out = jnp.transpose(out, (0, 2, 1))  # [heads, head_dim, N]
        # Two-step reshape to avoid ShardingTypeError with multi-device
        # (split spatial axis first, then merge head axes)
        out = jnp.reshape(out, (self.num_heads, self.head_dim, H, W))  # [heads, head_dim, H, W]
        out = jnp.reshape(out, (C, H, W))

        out = self.to_out(out)
        return out + residual


class UpsampledConv2d(eqx.Module):
    """Nearest-neighbor upsample followed by Conv2d."""
    conv: nn.Conv2d
    scale: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale: int = 2,
        padding: int = 1,
        key=None,
    ):
        self.scale = scale
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            key=key,
        )

    def __call__(self, x):
        # x shape: [C, H, W]
        c, h, w = x.shape
        upsampled_size = (c, h * self.scale, w * self.scale)
        upsampled = jax.image.resize(x, upsampled_size, method="nearest")
        return self.conv(upsampled)


class Encoder2d(eqx.Module):
    """Encoder: 256 -> 128 -> 64 -> 32 -> 16 (total 16x downsampling).

    Configurable capacity with channel multipliers, ResBlocks per stage, and optional attention.
    """
    # Stage convs and res blocks stored as tuples
    in_conv: nn.Conv2d
    stages: tuple  # tuple of (downsample_conv, res_blocks tuple)
    bottleneck_res: tuple  # ResBlocks at bottleneck
    bottleneck_attn: tp.Optional[SelfAttention2d]
    proj: nn.Conv2d
    norm_out: tp.Optional[nn.GroupNorm]

    use_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int = 512,
        codebook_dim: int = 64,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        use_attention: bool = True,
        use_norm: bool = True,
        attention_heads: int = 8,
        in_channels: int = 1,
        key=None,
    ):
        """
        Args:
            hidden_dim: Channels at bottleneck (before projection)
            codebook_dim: Output channels (latent dimension)
            base_channels: Base channel count, multiplied by channel_mult
            channel_mult: Channel multipliers for each downsampling stage
            num_res_blocks: Number of ResBlocks per resolution stage
            use_attention: Whether to use self-attention at bottleneck
            use_norm: Whether to use GroupNorm in ResBlocks
            attention_heads: Number of attention heads
            in_channels: Number of input channels
        """
        self.use_norm = use_norm
        num_stages = len(channel_mult)
        total_blocks = 1 + num_stages * (1 + num_res_blocks) + num_res_blocks + 2
        keys = iter(jax.random.split(key, total_blocks + 5))

        # Initial conv: in_channels -> base_channels
        ch_in = base_channels * channel_mult[0]
        self.in_conv = nn.Conv2d(in_channels, ch_in, kernel_size=3, stride=2, padding=1, key=next(keys))

        # Build stages
        stages = []
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult
            # ResBlocks at this resolution
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append(ResBlock2d(ch_in, use_norm=use_norm, key=next(keys)))
            # Downsample to next resolution (except last stage)
            if i < num_stages - 1:
                ch_next = base_channels * channel_mult[i + 1]
                down_conv = nn.Conv2d(ch_in, ch_next, kernel_size=3, stride=2, padding=1, key=next(keys))
                stages.append((down_conv, tuple(res_blocks)))
                ch_in = ch_next
            else:
                stages.append((None, tuple(res_blocks)))
        self.stages = tuple(stages)

        # Bottleneck: project to hidden_dim, apply res blocks and optional attention
        ch_final = base_channels * channel_mult[-1]
        bottleneck_res = []
        for _ in range(num_res_blocks):
            bottleneck_res.append(ResBlock2d(ch_final, use_norm=use_norm, key=next(keys)))
        self.bottleneck_res = tuple(bottleneck_res)

        if use_attention:
            self.bottleneck_attn = SelfAttention2d(
                ch_final, num_heads=attention_heads, use_norm=use_norm, key=next(keys)
            )
        else:
            self.bottleneck_attn = None

        # Output norm and projection
        if use_norm:
            self.norm_out = nn.GroupNorm(groups=min(32, ch_final), channels=ch_final)
        else:
            self.norm_out = None
        self.proj = nn.Conv2d(ch_final, codebook_dim, kernel_size=1, key=next(keys))

    def __call__(self, x):
        y = self.in_conv(x)

        for down_conv, res_blocks in self.stages:
            for res_block in res_blocks:
                y = res_block(y)
            if down_conv is not None:
                y = down_conv(jax.nn.silu(y))

        for res_block in self.bottleneck_res:
            y = res_block(y)

        if self.bottleneck_attn is not None:
            y = self.bottleneck_attn(y)

        if self.norm_out is not None:
            y = self.norm_out(y)
        y = jax.nn.silu(y)
        y = self.proj(y)
        return y


class Decoder2d(eqx.Module):
    """Decoder: 16 -> 32 -> 64 -> 128 -> 256 (total 16x upsampling).

    Configurable capacity with channel multipliers, ResBlocks per stage, and optional attention.
    """
    proj: nn.Conv2d
    bottleneck_attn: tp.Optional[SelfAttention2d]
    bottleneck_res: tuple
    stages: tuple  # tuple of (upsample_conv, res_blocks tuple)
    final_up: UpsampledConv2d  # Final 2x upsample to match input resolution
    final_res: tuple
    norm_out: tp.Optional[nn.GroupNorm]
    out_conv: nn.Conv2d

    use_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int = 512,
        codebook_dim: int = 64,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        use_attention: bool = True,
        use_norm: bool = True,
        attention_heads: int = 8,
        out_channels: int = 1,
        key=None,
    ):
        """
        Args:
            hidden_dim: Channels at bottleneck (after projection)
            codebook_dim: Input channels (latent dimension)
            base_channels: Base channel count, multiplied by channel_mult
            channel_mult: Channel multipliers (reversed for decoder)
            num_res_blocks: Number of ResBlocks per resolution stage
            use_attention: Whether to use self-attention at bottleneck
            use_norm: Whether to use GroupNorm in ResBlocks
            attention_heads: Number of attention heads
            out_channels: Number of output channels
        """
        self.use_norm = use_norm
        num_stages = len(channel_mult)
        # Reverse channel_mult for decoder (going from small to large resolution)
        dec_mult = channel_mult[::-1]
        total_blocks = num_stages * (1 + num_res_blocks) + num_res_blocks * 2 + 10
        keys = iter(jax.random.split(key, total_blocks + 10))

        # Project from codebook_dim to bottleneck channels
        ch_in = base_channels * dec_mult[0]
        self.proj = nn.Conv2d(codebook_dim, ch_in, kernel_size=1, key=next(keys))

        # Bottleneck attention and res blocks
        if use_attention:
            self.bottleneck_attn = SelfAttention2d(
                ch_in, num_heads=attention_heads, use_norm=use_norm, key=next(keys)
            )
        else:
            self.bottleneck_attn = None

        bottleneck_res = []
        for _ in range(num_res_blocks):
            bottleneck_res.append(ResBlock2d(ch_in, use_norm=use_norm, key=next(keys)))
        self.bottleneck_res = tuple(bottleneck_res)

        # Build upsampling stages (num_stages - 1 upsamplings between stages)
        stages = []
        for i, mult in enumerate(dec_mult):
            # ResBlocks at this resolution
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append(ResBlock2d(ch_in, use_norm=use_norm, key=next(keys)))
            # Upsample to next resolution (except last stage)
            if i < num_stages - 1:
                ch_next = base_channels * dec_mult[i + 1]
                up_conv = UpsampledConv2d(ch_in, ch_next, key=next(keys))
                stages.append((up_conv, tuple(res_blocks)))
                ch_in = ch_next
            else:
                stages.append((None, tuple(res_blocks)))
        self.stages = tuple(stages)

        # Final upsampling to match input resolution (encoder does in_conv with stride=2)
        ch_final = base_channels * dec_mult[-1]
        self.final_up = UpsampledConv2d(ch_final, ch_final, key=next(keys))
        final_res = []
        for _ in range(num_res_blocks):
            final_res.append(ResBlock2d(ch_final, use_norm=use_norm, key=next(keys)))
        self.final_res = tuple(final_res)

        # Output norm and conv
        if use_norm:
            self.norm_out = nn.GroupNorm(groups=min(32, ch_final), channels=ch_final)
        else:
            self.norm_out = None
        self.out_conv = nn.Conv2d(ch_final, out_channels, kernel_size=3, padding=1, key=next(keys))

    def __call__(self, x):
        y = self.proj(x)

        if self.bottleneck_attn is not None:
            y = self.bottleneck_attn(y)

        for res_block in self.bottleneck_res:
            y = res_block(y)

        for up_conv, res_blocks in self.stages:
            for res_block in res_blocks:
                y = res_block(y)
            if up_conv is not None:
                y = up_conv(jax.nn.silu(y))

        # Final upsample to full resolution
        y = self.final_up(jax.nn.silu(y))
        for res_block in self.final_res:
            y = res_block(y)

        if self.norm_out is not None:
            y = self.norm_out(y)
        y = jax.nn.silu(y)
        y = self.out_conv(y)
        return y


class MultiScaleQuantizer2d(eqx.Module):
    """Multi-scale residual quantizer for VAR-style VQ-VAE.

    Progressively quantizes residuals at increasing resolutions (e.g., 1x1 → 2x2 → 4x4 → 8x8 → 16x16).
    Each scale has its own independent codebook, capturing scale-specific detail.
    Scales are specified as (h, w) tuples to support non-square latents.
    """
    scales: tuple = eqx.field(static=True)  # e.g., ((1,1), (2,2), (4,4), (8,8), (16,16))
    target_h: int = eqx.field(static=True)  # target height (max of scale heights)
    target_w: int = eqx.field(static=True)  # target width (max of scale widths)
    K: int = eqx.field(static=True)  # vocab_size
    D: int = eqx.field(static=True)  # codebook_dim

    codebooks: tuple  # tuple of n_scales arrays, each [K, D]
    phi_convs: tuple  # per-scale 3x3 convolutions
    codebook_avgs: tuple  # tuple of n_scales arrays, each [K, D]
    cluster_sizes: tuple  # tuple of n_scales arrays, each [K]

    decay: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        scales: tuple = ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16)),
        vocab_size: int = 4096,
        codebook_dim: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        key=None,
    ):
        self.scales = scales
        self.target_h = max(s[0] for s in scales)
        self.target_w = max(s[1] for s in scales)
        self.K = vocab_size
        self.D = codebook_dim
        self.decay = decay
        self.eps = eps

        n_scales = len(scales)
        # n_scales keys for codebooks + n_scales keys for phi_convs
        keys = jax.random.split(key, 2 * n_scales)

        # Per-scale codebooks
        codebooks = []
        codebook_avgs = []
        cluster_sizes = []
        for k in range(n_scales):
            cb = jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            )(keys[k], (vocab_size, codebook_dim))
            codebooks.append(cb)
            codebook_avgs.append(jnp.copy(cb))
            cluster_sizes.append(jnp.zeros(vocab_size))
        self.codebooks = tuple(codebooks)
        self.codebook_avgs = tuple(codebook_avgs)
        self.cluster_sizes = tuple(cluster_sizes)

        # Per-scale convolutions
        phi_convs = []
        for i in range(n_scales):
            conv = nn.Conv2d(codebook_dim, codebook_dim, kernel_size=3, padding=1, key=keys[n_scales + i])
            phi_convs.append(conv)
        self.phi_convs = tuple(phi_convs)

    def __call__(self, x):
        """Multi-scale quantization with residual learning.

        Args:
            x: Encoder output [D, H, W] where H=target_h, W=target_w

        Returns:
            z_q: Quantized output [D, H, W]
            codebook_updates: Tuple of n_scales tuples, each (cluster_size, codebook_avg, codebook)
            indices_list: List of indices for each scale
            commit_loss: Per-scale commitment loss (sum of MSE at each scale)
        """
        D, H, W = x.shape
        assert H == self.target_h and W == self.target_w, f"Expected {self.target_h}x{self.target_w}, got {H}x{W}"

        r = x  # residual = encoder output
        quantized_sum = jnp.zeros_like(x)
        indices_list = []
        per_scale_flattens = []
        per_scale_indices = []
        commit_loss = 0.0

        for k, (sh, sw) in enumerate(self.scales):
            # Downsample residual to current scale
            r_k = jax.image.resize(r, (D, sh, sw), method="bilinear")

            # Per-scale convolution
            r_k = self.phi_convs[k](r_k)

            # Quantize at this scale using that scale's codebook
            z_q_k, flatten_k, indices_k = self._quantize_single(r_k, self.codebooks[k])

            # Per-scale commitment loss: encourage r_k to be close to quantized z_q_k
            commit_loss = commit_loss + jnp.mean(
                (r_k - jax.lax.stop_gradient(z_q_k)) ** 2
            )

            # Upsample quantized back to target size
            z_q_k_up = jax.image.resize(z_q_k, (D, self.target_h, self.target_w), method="nearest")

            # Accumulate and compute residual for next scale
            quantized_sum = quantized_sum + z_q_k_up
            r = x - quantized_sum  # residual for next scale

            indices_list.append(indices_k)
            per_scale_flattens.append(flatten_k)
            per_scale_indices.append(indices_k.flatten())

        # Compute independent codebook updates per scale
        codebook_updates = self._per_scale_codebook_updates(per_scale_flattens, per_scale_indices)

        return quantized_sum, codebook_updates, indices_list, commit_loss

    def _quantize_single(self, x, codebook):
        """Quantize a single-scale feature map.

        Args:
            x: Feature map [D, H, W]
            codebook: Codebook array [K, D] for this scale

        Returns:
            z_q: Quantized [D, H, W]
            flatten: Flattened features [H*W, D]
            indices: Codebook indices [H, W]
        """
        D, H, W = x.shape

        # Flatten: [D, H, W] -> [H*W, D]
        flatten = jnp.transpose(x, (1, 2, 0))  # [H, W, D]
        flatten = jnp.reshape(flatten, (-1, self.D))  # [H*W, D]

        # L2-normalize before computing distances (stabilizes commitment loss)
        flatten = flatten / (jnp.linalg.norm(flatten, axis=-1, keepdims=True) + 1e-8)
        codebook = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)

        # Compute distances to codebook vectors
        distance = (
            2.0 - 2.0 * jnp.matmul(flatten, jnp.transpose(codebook))
        )

        # Find nearest codebook vectors
        codebook_indices = jnp.argmin(distance, axis=-1)  # [H*W]
        # Use one-hot matmul instead of fancy indexing for multi-device sharding
        z_q = jax.nn.one_hot(codebook_indices, codebook.shape[0]) @ codebook  # [H*W, D]

        # Straight-through estimator
        z_q = flatten + jax.lax.stop_gradient(z_q - flatten)

        # Reshape back: [H*W, D] -> [D, H, W]
        z_q = jnp.reshape(z_q, (H, W, self.D))  # [H, W, D]
        z_q = jnp.transpose(z_q, (2, 0, 1))  # [D, H, W]

        # Reshape indices: [H*W] -> [H, W]
        indices = jnp.reshape(codebook_indices, (H, W))

        return z_q, flatten, indices

    def _per_scale_codebook_updates(self, per_scale_flattens, per_scale_indices):
        """Compute independent EMA updates for each scale's codebook.

        Returns:
            Tuple of n_scales tuples, each (cluster_size, codebook_avg, codebook).
        """
        all_updates = []
        for k in range(len(self.scales)):
            flatten_k = per_scale_flattens[k]
            indices_k = per_scale_indices[k]

            # Calculate usage for this scale
            codebook_onehot = jax.nn.one_hot(indices_k, self.K)
            codebook_onehot_sum = jnp.sum(codebook_onehot, axis=0)
            codebook_sum = jnp.dot(flatten_k.T, codebook_onehot)

            # EMA updates using this scale's state
            new_cluster_size = (
                self.decay * self.cluster_sizes[k] + (1 - self.decay) * codebook_onehot_sum
            )
            new_codebook_avg = (
                self.decay * self.codebook_avgs[k] + (1 - self.decay) * codebook_sum.T
            )

            # Laplace smoothing and normalization
            n = jnp.sum(new_cluster_size)
            new_cluster_size_smooth = (new_cluster_size + self.eps) / (n + self.K * self.eps) * n
            new_codebook = new_codebook_avg / new_cluster_size_smooth[:, None]

            all_updates.append((new_cluster_size, new_codebook_avg, new_codebook))

        return tuple(all_updates)


class VQVAE2d(eqx.Module):
    """Multi-scale VQ-VAE for 2D turbulence data."""
    encoder: Encoder2d
    decoder: Decoder2d
    quantizer: MultiScaleQuantizer2d

    def __init__(
        self,
        hidden_dim: int = 512,
        codebook_dim: int = 64,
        vocab_size: int = 4096,
        scales: tuple = ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16)),
        decay: float = 0.99,
        # Encoder/decoder capacity parameters
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        use_attention: bool = True,
        use_norm: bool = True,
        attention_heads: int = 8,
        in_channels: int = 1,
        key=None,
    ):
        """
        Args:
            hidden_dim: Legacy param, unused with new architecture
            codebook_dim: Latent dimension per spatial position
            vocab_size: Number of codebook vectors
            scales: Multi-scale quantization resolutions as (h, w) tuples
            decay: EMA decay for codebook updates
            base_channels: Base channel count for encoder/decoder
            channel_mult: Channel multipliers per stage (e.g., (1,2,4,4) -> 128,256,512,512)
            num_res_blocks: ResBlocks per resolution stage
            use_attention: Enable self-attention at bottleneck
            use_norm: Enable GroupNorm in ResBlocks
            attention_heads: Number of attention heads
            in_channels: Number of input/output channels
        """
        key1, key2, key3 = jax.random.split(key, 3)

        self.encoder = Encoder2d(
            hidden_dim=hidden_dim,
            codebook_dim=codebook_dim,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            use_norm=use_norm,
            attention_heads=attention_heads,
            in_channels=in_channels,
            key=key1,
        )
        self.decoder = Decoder2d(
            hidden_dim=hidden_dim,
            codebook_dim=codebook_dim,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            use_norm=use_norm,
            attention_heads=attention_heads,
            out_channels=in_channels,
            key=key2,
        )
        self.quantizer = MultiScaleQuantizer2d(
            scales=scales,
            vocab_size=vocab_size,
            codebook_dim=codebook_dim,
            decay=decay,
            key=key3,
        )

    def __call__(self, x):
        """Forward pass through multi-scale VQ-VAE.

        Args:
            x: Input [C, H, W]

        Returns:
            z_e: Encoder output [D, H', W']
            z_q: Quantized latent [D, H', W']
            codebook_updates: Tuple for EMA updates
            indices_list: List of indices for each scale
            commit_loss: Per-scale commitment loss from quantizer
            y: Reconstruction [C, H, W]
        """
        z_e = self.encoder(x)  # [codebook_dim, 16, 16]
        z_q, codebook_updates, indices_list, commit_loss = self.quantizer(z_e)
        y = self.decoder(z_q)  # [1, H, W]
        return z_e, z_q, codebook_updates, indices_list, commit_loss, y

    def encode(self, x):
        """Encode input to quantized latent and indices."""
        z_e = self.encoder(x)
        z_q, _, indices_list, _ = self.quantizer(z_e)
        return z_q, indices_list

    def decode(self, z_q):
        """Decode from quantized latent."""
        return self.decoder(z_q)

    def decode_indices(self, indices_list):
        """Decode from multi-scale codebook indices.

        Args:
            indices_list: List of indices for each scale, shapes [sh, sw] for (sh, sw) in scales

        Returns:
            Reconstruction [C, H, W]
        """
        D = self.quantizer.D
        target_h = self.quantizer.target_h
        target_w = self.quantizer.target_w
        z_q = jnp.zeros((D, target_h, target_w))

        for k, indices in enumerate(indices_list):
            H, W = indices.shape
            z_q_k = jax.nn.one_hot(indices.flatten(), self.quantizer.K) @ self.quantizer.codebooks[k]  # [H*W, D]
            z_q_k = jnp.reshape(z_q_k, (H, W, D))  # [H, W, D]
            z_q_k = jnp.transpose(z_q_k, (2, 0, 1))  # [D, H, W]
            z_q_k_up = jax.image.resize(z_q_k, (D, target_h, target_w), method="nearest")
            z_q = z_q + z_q_k_up

        return self.decoder(z_q)


