import pytest
import jax
import jax.numpy as jnp

@pytest.fixture
def prng_key():
    """Provides a deterministic JAX random key for reproducible tests."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def dummy_1d_x():
    return jnp.array([0.5])

@pytest.fixture
def dummy_2d_x():
    return jnp.array([0.5, -1.2])

@pytest.fixture
def dummy_5d_x():
    return jnp.array([0.1, -0.5, 1.2, 0.0, -2.1])