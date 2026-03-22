'''
Shared settings for ZDT multi-objective examples (NSGA-II, NSGA-III, DE).

Ensures identical initial populations (via synchronized NumPy seed before
`initialize_population`), aligned random streams for GA operators, and
consistent objective-space axis limits across figures.
'''

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np

MASTER_SEED = 42
N_INPUT = 3
POPULATION_SIZE = 32
MAX_ITERATIONS = 20

# Objective-space limits: f1 shared; f2 depends on ZDT (same across NSGA-II / III / DE figures)
PLOT_F1_LIM: Tuple[float, float] = (-0.1, 1.1)
PLOT_F2_LIM_BY_BENCHMARK: Dict[str, Tuple[float, float]] = {
    "ZDT1": (-1.0, 10.0),
    "ZDT2": (-1.0, 10.0),
    "ZDT3": (-1.0, 10.0),
    "ZDT4": (-5.0, 45.0),
    "ZDT6": (-1.0, 10.0),
}


def apply_benchmark_seeds(bench_index: int) -> None:
    '''
    Seed NumPy global RNG and Python `random` for one benchmark run.

    `bench_index` is the 0-based index in the shared benchmark list.

    DE uses a separate NumPy ``Generator`` via ``de_rng_seed`` because ``OptDE``
    otherwise calls ``default_rng()`` with no seed each iteration.
    '''
    s = MASTER_SEED + bench_index * 1_000_000
    np.random.seed(s)
    random.seed(s)


def de_rng_seed(bench_index: int) -> int:
    '''Deterministic seed for DE/rand/1/bin (independent of post-init NumPy state).'''
    return MASTER_SEED + bench_index * 1_000_000 + 50_000_001
