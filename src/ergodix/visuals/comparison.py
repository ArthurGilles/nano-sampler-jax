"""Compare several sample sets side by side with a shared time clock."""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from matplotlib import animation

from ..distributions import TargetDistribution
from ._core import _Panel, _particles_to_time_major, _render, _to_time_major

PDF = Union[TargetDistribution, Callable]


def _broadcast(value, n: int, name: str) -> List:
    """Expand ``value`` into a length-``n`` list of per-set pdfs."""
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"`{name}` must have one entry per sample set ({n}); got {len(value)}."
            )
        return list(value)
    return [value] * n  # a single pdf shared by every set


def _derive_particles_path(save_path: str) -> str:
    """Insert ``_particles`` before the suffix of ``save_path``."""
    path = Path(save_path)
    return str(path.with_name(f"{path.stem}_particles{path.suffix}"))


def compare_samples(
    sample_sets: Sequence,
    *,
    pdfs: Optional[Union[PDF, Sequence[Optional[PDF]]]] = None,
    particles_pdfs: Optional[Union[PDF, Sequence[Optional[PDF]]]] = None,
    labels: Optional[Sequence[str]] = None,
    pdf_is_log: bool = True,
    time_grid=None,
    bins: int = 60,
    levels: int = 8,
    grid_size: int = 200,
    fps: int = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[animation.FuncAnimation, Optional[animation.FuncAnimation]]:
    """
    Compare several sets of samples (or ``(samples, particles)`` pairs) at once.

    All sample sets are animated side by side in one figure, driven by a single
    frame clock so time advances at the same speed for every set. Sets that come
    with particles get a second, separate figure laying their particle clouds out
    in the same style. This is convenient for comparing, e.g., SLIPS runs with
    different hyper-parameters against the same target.

    Parameters
    ----------
        sample_sets: sequence
            One entry per set. Each entry is either a ``samples`` array of shape
            ``(n_samples, n_steps, dim)`` (the SLIPS ``Y_hist``) or a
            ``(samples, particles)`` pair where ``particles`` is the SLIPS
            ``X_hist``. Every set must share the same ``dim`` in ``{1, 2}`` and
            the same ``n_steps``.
        pdfs: TargetDistribution or Callable or sequence, optional
            A single density shared by every samples panel, or one per set
            (use ``None`` to skip a panel). Interpreted as a log-density unless
            ``pdf_is_log=False`` and normalised to mass 1.
        particles_pdfs: TargetDistribution or Callable or sequence, optional
            Same as ``pdfs`` but for the particle panels.
        labels: sequence of str, optional
            Title for each set; defaults to ``"set 1"``, ``"set 2"``, ...
        pdf_is_log: bool, default True
            Whether the supplied pdfs return log-densities.
        time_grid: jax.Array, optional
            Time points used to label the frames.
        bins, levels, grid_size, fps, figsize
            See :func:`ergodix.visuals.animate_samples`.
        save_path: str, optional
            If given, the samples figure is written here and the particles figure
            to a sibling path with ``_particles`` inserted before the suffix
            (e.g. ``out.gif`` -> ``out_particles.gif``).
    Returns
    -------
    samples_anim: matplotlib.animation.FuncAnimation
        Animation comparing the samples.
    particles_anim: matplotlib.animation.FuncAnimation or None
        Animation comparing the particles, or ``None`` if no set provided any.

    Example
    -------
    >>> from ergodix.visuals import compare_samples
    >>> samples_anim, particles_anim = compare_samples(
    ...     [(y_hist_a, x_hist_a), (y_hist_b, x_hist_b)],
    ...     pdfs=target, labels=["run A", "run B"], save_path="compare.gif",
    ... )
    """
    sets = list(sample_sets)
    n_sets = len(sets)
    if n_sets == 0:
        raise ValueError("`sample_sets` must contain at least one set.")

    pdf_list = _broadcast(pdfs, n_sets, "pdfs")
    particles_pdf_list = _broadcast(particles_pdfs, n_sets, "particles_pdfs")
    if labels is None:
        labels = [f"set {i + 1}" for i in range(n_sets)]
    elif len(labels) != n_sets:
        raise ValueError(
            f"`labels` must have one entry per sample set ({n_sets}); got {len(labels)}."
        )

    sample_panels: List[_Panel] = []
    particle_panels: List[_Panel] = []
    for i, item in enumerate(sets):
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(
                    f"`sample_sets[{i}]` must be a `samples` array or a "
                    f"`(samples, particles)` pair; got a sequence of length {len(item)}."
                )
            raw_samples, raw_particles = item
        else:
            raw_samples, raw_particles = item, None

        sample_frames = _to_time_major(raw_samples, name=f"sample_sets[{i}]")
        sample_panels.append(
            _Panel(sample_frames, labels[i], pdf_list[i], pdf_is_log)
        )
        if raw_particles is not None:
            particle_frames = _particles_to_time_major(raw_particles)
            particle_panels.append(
                _Panel(particle_frames, labels[i], particles_pdf_list[i], pdf_is_log)
            )

    _, samples_anim = _render(
        sample_panels,
        fps=fps,
        save_path=save_path,
        figsize=figsize,
        time_grid=time_grid,
        bins=bins,
        levels=levels,
        grid_size=grid_size,
    )

    particles_anim = None
    if particle_panels:
        particles_save = _derive_particles_path(save_path) if save_path else None
        _, particles_anim = _render(
            particle_panels,
            fps=fps,
            save_path=particles_save,
            figsize=figsize,
            time_grid=time_grid,
            bins=bins,
            levels=levels,
            grid_size=grid_size,
        )

    return samples_anim, particles_anim
