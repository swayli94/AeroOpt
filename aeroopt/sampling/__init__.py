'''
Design-of-experiments samplers on the unit hypercube.

All samplers return values in `[0, 1]^n_dim`; mapping to the problem bounds is
done by the caller (see :meth:`aeroopt.core.problem.Problem.latin_hypercube_sampling`).
'''
from aeroopt.sampling.doe import (
    latin_hypercube_sampling,
    random_sampling,
)

__all__ = [
    'latin_hypercube_sampling',
    'random_sampling',
]
