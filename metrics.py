import jax
import jax.numpy as jnp


@jax.jit
def kullback_leibler_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Computes the sample KLD between two inputs.
    
    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.

    TODO: add an eps=1e-32 to remove zero issues?
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    eps=1e-12

    kld = (p + eps) * jnp.log((p + eps)  / (q + eps))
    return jnp.sum(kld, axis=-2)


@jax.jit
def KLDLoss(p: jnp.ndarray, q: jnp.ndarray):
    """Reduce mapped KLD to loss value."""
    return jnp.mean(kullback_leibler_divergence(p, q))


@jax.jit
def jensen_shannon_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Computes the sample JSD between two inputs.
    
    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    return (kullback_leibler_divergence(p, q) + kullback_leibler_divergence(q, p)) / 2


@jax.jit
def JSDLoss(p: jnp.ndarray, q: jnp.ndarray):
    """Reduce mapped JSD to loss value."""
    return jnp.mean(jensen_shannon_divergence(p, q))
