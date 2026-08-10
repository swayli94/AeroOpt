'''
Design-of-experiments samplers on the unit hypercube `[0, 1]^n_dim`.
'''

from __future__ import annotations

import numpy as np


def random_sampling(n_dim: int, n_sample: int,
                    seed: int | np.random.Generator | None = None) -> np.ndarray:
    '''
    Uniform random sampling on the unit hypercube.

    Parameters
    ----------
    n_dim: int
        Number of dimensions.
    n_sample: int
        Number of samples.
    seed: int, numpy.random.Generator, or None
        Seed or generator for reproducible sampling.

    Returns
    -------
    samples: np.ndarray [n_sample, n_dim]
        Samples in `[0, 1]`.
    '''
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.random((int(n_sample), int(n_dim)))


def latin_hypercube_sampling(n_dim: int, n_sample: int,
                             criterion: str | None = 'm',
                             seed: int | np.random.Generator | None = None) -> np.ndarray:
    '''
    Latin hypercube sampling on the unit hypercube.

    Each of the `n_dim` axes is split into `n_sample` equal-probability strata and
    every stratum receives exactly one sample, which gives far better space filling
    than plain random sampling at the same budget. The `'m'` (maximin) criterion
    additionally permutes the strata to maximise the minimum pairwise distance.

    This delegates to :func:`pydoe.lhs`.

    Parameters
    ----------
    n_dim: int
        Number of dimensions.
    n_sample: int
        Number of samples.
    criterion: str or None
        Space-filling criterion passed to `pydoe.lhs`, e.g. `'m'` (maximin),
        `'c'` (centered), `'cm'`, `'corr'`. `None` uses plain stratified sampling.
    seed: int, numpy.random.Generator, or None
        Seed or generator for reproducible sampling.

    Returns
    -------
    samples: np.ndarray [n_sample, n_dim]
        Samples in `[0, 1]`.
    '''
    import pydoe

    samples = pydoe.lhs(int(n_dim), samples=int(n_sample), criterion=criterion, seed=seed)

    return np.asarray(samples, dtype=float)
