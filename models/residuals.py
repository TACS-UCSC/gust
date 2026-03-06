import jax
import jax.numpy as jnp

# --- 1. Setup Dummy Dimensions ---
key = jax.random.PRNGKey(42)
D = 16              # Hidden dimension
vocab_size = 50     # Vocabulary size

# --- 2. Build the Input Sequence ---
# Indices 0-3: Scale 2 Embeddings (4 tokens)
# Indices 4-19: Scale 3 Embeddings (16 tokens - The Answer Key!)
seq_emb = jax.random.normal(key, (20, D))

# The actual ground truth integer targets for Scale 3
targets_S3 = jax.random.randint(key, (16,), 0, vocab_size)

# Dummy weights for our 1-to-4 Expansion Head
# Maps (D) -> (4 * vocab_size)
W_head = jax.random.normal(key, (D, 4 * vocab_size))

# --- 3. The Leaky Transformer + Shifted Prediction ---
def predict_and_compute_loss(x):
    # A. The Causal Mask
    # 1 means "allowed to attend", 0 means "blocked"
    mask = jnp.tril(jnp.ones((20, 20))) 
    
    # B. The Attention Mechanism
    attn_scores = x @ x.T
    attn_scores = jnp.where(mask == 1, attn_scores, -1e9)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    attn_out = attn_weights @ x
    
    # C. THE RESIDUAL CONNECTION (The suspected leak!)
    h = x + attn_out 
    
    # D. The Shifted Extraction (VAR Magic)
    # We want to predict Scale 3 (16 tokens).
    # We slice out the hidden states at the SCALE 2 positions (Indices 0 to 3)
    h_source = h[0:4] 
    
    # E. The 1-to-4 Expansion
    logits_expanded = h_source @ W_head        # Shape: (4, 4 * vocab_size)
    logits = logits_expanded.reshape(16, 50)   # Shape: (16, vocab_size)
    
    # F. The Loss
    # Compare our 16 predictions against the 16 Scale 3 ground truth targets
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets_S3[:, None], axis=-1).squeeze(-1)
    loss = -jnp.mean(target_log_probs)
    
    return loss

# --- 4. The Gradient Test ---
# We ask JAX: "How much did each input token contribute to this loss?"
grad_fn = jax.grad(predict_and_compute_loss)
gradients = grad_fn(seq_emb)

# Calculate the magnitude of the gradients at the different sequence positions
grad_norm_S2 = jnp.linalg.norm(gradients[0:4])   # The Scale 2 positions
grad_norm_S3 = jnp.linalg.norm(gradients[4:20])  # The Scale 3 positions

print(f"Gradient flowing into Scale 2 Input (Positions 0-3):  {grad_norm_S2:.6f}")
print(f"Gradient flowing into Scale 3 Input (Positions 4-19): {grad_norm_S3:.6f}")

# The ultimate mathematical assertion:
assert grad_norm_S3 == 0.0, "LEAK DETECTED! The answer key reached the output."
print("\nAssertion passed: ZERO gradients reached the Scale 3 inputs.")