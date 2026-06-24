"""Project N-D samples onto a line or plane, then animate them over time."""

from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from matplotlib import animation

from ..distributions import TargetDistribution
from ._core import _Panel, _particles_to_time_major, _render, _to_time_major

PDF = Union[TargetDistribution, Callable]


def _make_basis(directions: jnp.ndarray, orthonormalize: bool) -> np.ndarray:
    """Turn one or two direction vectors into an orthonormal projection basis.

    Returns an array of shape ``(p, dim)`` (``p`` in ``{1, 2}``) whose rows are
    unit vectors. With two directions and ``orthonormalize=True`` the second is
    made orthogonal to the first (Gram-Schmidt) so the pair spans a proper plane.
    """
    dirs = np.asarray(directions, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    if dirs.ndim != 2 or dirs.shape[0] not in (1, 2):
        raise ValueError(
            "`directions` must be a single vector of shape (dim,) or two vectors "
            f"of shape (2, dim); got shape {dirs.shape}."
        )

    norm0 = np.linalg.norm(dirs[0])
    if norm0 == 0.0:
        raise ValueError("`directions` contains a zero vector; cannot project.")
    u0 = dirs[0] / norm0
    if dirs.shape[0] == 1:
        return u0[None, :]

    v1 = dirs[1]
    if orthonormalize:
        v1 = v1 - np.dot(v1, u0) * u0
    norm1 = np.linalg.norm(v1)
    if norm1 == 0.0:
        raise ValueError(
            "The two `directions` are collinear; they cannot span a plane. "
            "Provide two linearly independent vectors."
        )
    u1 = v1 / norm1
    return np.stack([u0, u1], axis=0)


def _project(frames: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project time-major frames ``(n_steps, n, dim)`` onto ``basis`` rows."""
    return frames @ basis.T


def _slice_pdf(pdf: Callable, basis: np.ndarray, origin: np.ndarray) -> Callable:
    """Restrict ``pdf`` to the affine subspace ``origin + s @ basis``.

    The returned callable maps a projected coordinate ``s`` of shape ``(p,)`` to
    ``pdf(origin + s @ basis)``, i.e. a **slice** of the density through the
    chosen subspace (not a marginal, which would require integrating out the
    remaining dimensions).
    """
    basis_j = jnp.asarray(basis)
    origin_j = jnp.asarray(origin)

    def sliced(s: jnp.ndarray) -> jnp.ndarray:
        return pdf(origin_j + s @ basis_j)

    return sliced


def animate_projection(
    samples,
    directions,
    *,
    pdf: Optional[PDF] = None,
    particles=None,
    particles_pdf: Optional[PDF] = None,
    pdf_is_log: bool = True,
    origin=None,
    orthonormalize: bool = True,
    time_grid=None,
    bins: int = 60,
    levels: int = 8,
    grid_size: int = 200,
    fps: int = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Tuple[str, str] = ("Samples", "Particles"),
) -> animation.FuncAnimation:
    """
    Animate N-dimensional samples projected onto a line or a plane.

    The N-D samples are projected onto the subspace spanned by ``directions``
    (one vector for a 1D view, two for a 2D view) and then animated exactly like
    :func:`ergodix.visuals.animate_samples`. This makes it possible to inspect
    high-dimensional SLIPS output along chosen directions.

    Parameters
    ----------
        samples: jax.Array
            Samples over time of shape ``(n_samples, n_steps, dim)`` (the SLIPS
            ``Y_hist``), with ``dim`` arbitrary.
        directions: jax.Array
            Projection direction(s). Shape ``(dim,)`` projects onto a line (1D
            view); shape ``(2, dim)`` projects onto a plane (2D view). Vectors
            need not be unit length - they are normalised internally.
        pdf: TargetDistribution or Callable, optional
            Density overlaid on the projected samples, evaluated as a **slice**
            through the projection subspace, ``s -> pdf(origin + s @ basis)``, and
            normalised to mass 1 on the grid. Interpreted as a log-density unless
            ``pdf_is_log=False``. Note this is a slice, not a marginal.
        particles: jax.Array, optional
            Particle history (SLIPS ``X_hist`` of shape
            ``(n_samples, n_steps, n_chains, dim)`` or flat
            ``(n_samples, n_steps, dim)``), projected and drawn in a second panel.
        particles_pdf: TargetDistribution or Callable, optional
            Optional density slice overlaid on the particle panel.
        pdf_is_log: bool, default True
            Whether ``pdf`` and ``particles_pdf`` return log-densities.
        origin: jax.Array, optional
            Point of the full space the projection subspace passes through, used
            for the pdf slice. Defaults to the origin (zeros).
        orthonormalize: bool, default True
            For a 2D view, make the second direction orthogonal to the first so
            the pair spans a proper plane.
        time_grid: jax.Array, optional
            Time points used to label the frames.
        bins, levels, grid_size, fps, save_path, figsize, titles
            See :func:`ergodix.visuals.animate_samples`.
    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        The animation of the projected samples (and particles).

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from ergodix.visuals import animate_projection
    >>> # `y_hist` has shape (n_samples, n_steps, 10); look along two axes.
    >>> directions = jnp.stack([jnp.eye(10)[0], jnp.eye(10)[1]])
    >>> anim = animate_projection(y_hist, directions, pdf=target)
    """
    basis = _make_basis(jnp.asarray(directions), orthonormalize)
    proj_dim = basis.shape[1]

    sample_frames = _to_time_major(samples, name="samples")
    if sample_frames.shape[-1] != proj_dim:
        raise ValueError(
            f"`directions` dimension ({proj_dim}) must match `samples` dimension "
            f"({sample_frames.shape[-1]})."
        )

    origin_vec = np.zeros(proj_dim) if origin is None else np.asarray(origin, dtype=float)
    if origin_vec.shape != (proj_dim,):
        raise ValueError(
            f"`origin` must have shape ({proj_dim},); got {origin_vec.shape}."
        )

    sample_pdf = _slice_pdf(pdf, basis, origin_vec) if pdf is not None else None
    panels = [_Panel(_project(sample_frames, basis), titles[0], sample_pdf, pdf_is_log)]

    if particles is not None:
        particle_frames = _particles_to_time_major(particles)
        if particle_frames.shape[-1] != proj_dim:
            raise ValueError(
                f"`particles` dimension ({particle_frames.shape[-1]}) must match "
                f"`directions` dimension ({proj_dim})."
            )
        if particle_frames.shape[0] != sample_frames.shape[0]:
            raise ValueError(
                "`particles` and `samples` must have the same number of time steps."
            )
        part_pdf = (
            _slice_pdf(particles_pdf, basis, origin_vec)
            if particles_pdf is not None
            else None
        )
        panels.append(
            _Panel(_project(particle_frames, basis), titles[1], part_pdf, pdf_is_log)
        )

    _, anim = _render(
        panels,
        fps=fps,
        save_path=save_path,
        figsize=figsize,
        time_grid=time_grid,
        bins=bins,
        levels=levels,
        grid_size=grid_size,
    )
    return anim
