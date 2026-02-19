import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp


def update_codebook_ema(model, updates: tuple, indices_list: tp.List, key=None):
    """Update codebook using EMA for multi-scale model.

    Args:
        model: VQVAE2d model
        updates: Tuple of n_scales tuples, each (cluster_size, codebook_avg, codebook),
                 with a leading batch dimension from vmap.
        indices_list: List of per-scale indices (unused but kept for API compatibility)
        key: PRNG key for dead-code reinitialization
    """
    n_scales = len(model.quantizer.scales)
    keys = jax.random.split(key, n_scales)

    new_codebooks = []
    new_codebook_avgs = []
    new_cluster_sizes = []

    for k in range(n_scales):
        # Average over batch dimension for this scale's updates
        avg_k = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates[k])

        # Normalize cluster_size to get probability distribution
        n_total = jnp.sum(avg_k[0])
        h = avg_k[0] / n_total

        # Reinitialize codes that are under-used
        part_that_should_be = 1 / model.quantizer.K
        mask = (h < 0.25 * part_that_should_be)
        rand_embed = (
            jax.random.normal(keys[k], (model.quantizer.K, model.quantizer.D)) * mask[:, None]
        )

        target_size = n_total / model.quantizer.K
        cs = jnp.where(mask, target_size, avg_k[0])
        ca = jnp.where(mask[:, None], rand_embed * cs[:, None], avg_k[1])
        cb = jnp.where(mask[:, None], rand_embed, avg_k[2])

        new_cluster_sizes.append(cs)
        new_codebook_avgs.append(ca)
        new_codebooks.append(cb)

    def where(q):
        return q.quantizer.codebooks, q.quantizer.codebook_avgs, q.quantizer.cluster_sizes

    model = eqx.tree_at(where, model,
                         (tuple(new_codebooks), tuple(new_codebook_avgs), tuple(new_cluster_sizes)))
    return model


@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses(model, x, commitment_weight=0.1):
    """Calculate VQ-VAE losses for 2D data."""
    z_e, z_q, codebook_updates, indices_list, commit_loss_per_sample, y = jax.vmap(model)(x)

    reconstruct_loss = jnp.mean((x - y) ** 2)
    commit_loss = jnp.mean(commit_loss_per_sample)
    total_loss = reconstruct_loss + commitment_weight * commit_loss

    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices_list, y)


@eqx.filter_jit
def make_step(model, optimizer, opt_state, x, key, commitment_weight=0.1):
    """Single training step."""
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices_list, y)), grads = (
        calculate_losses(model, x, commitment_weight)
    )

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = update_codebook_ema(model, codebook_updates, indices_list, key)

    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        indices_list,
        y,
    )
