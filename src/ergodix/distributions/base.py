import jax
import equinox as eqx
from jaxtyping import Array, Float

class TargetDistribution(eqx.Module):
    """Base class for all target distributions."""

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        raise NotImplementedError

    def score(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """
        Score of the distribution, i.e. the gradient of the (unnormalized)
        log-density with respect to ``x``.

        The default implementation differentiates :meth:`__call__` with
        ``jax.grad``, so every distribution gets a correct score for free.
        Override it with a closed-form expression when one is available and
        cheaper than automatic differentiation.

        Parameters
        ----------
        x: Float[Array, "dim"]
            Point at which to evaluate the score, of shape ``(dim,)``.
        Returns
        -------
        Float[Array, "dim"]
            The score at ``x``, of shape ``(dim,)``.
        """
        return jax.grad(self.__call__)(x)