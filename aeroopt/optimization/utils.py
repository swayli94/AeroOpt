'''
Utility functions for optimization.
'''

from typing import List, Tuple

import numpy as np

from aeroopt.core import Database, Individual, Problem


def sbx_crossover(
        x1: np.ndarray, x2: np.ndarray, problem: Problem,
        cross_rate: float = 1.0, pow_sbx: float = 20.0,
        rng: np.random.Generator = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Simulated Binary Crossover (SBX) operator.

    Deb, Kalyanmoy, and Ram Bhushan Agrawal.
    "Simulated binary crossover for continuous search space."
    Complex systems 9.2 (1995): 115-148.
    '''
    if rng is None:
        rng = np.random.default_rng()

    child1 = x1.copy()
    child2 = x2.copy()
    if rng.random() > cross_rate:
        return child1, child2

    low = problem.data_settings.input_low
    upp = problem.data_settings.input_upp
    precision = problem.data_settings.input_precision

    for i in range(problem.n_input):
        if rng.random() > 0.5:
            continue
        if abs(x1[i] - x2[i]) <= precision[i]:
            continue

        y1, y2 = (x1[i], x2[i]) if x1[i] < x2[i] else (x2[i], x1[i])
        rand = rng.random()

        beta = 1.0 + (2.0 * (y1 - low[i]) / (y2 - y1))
        alpha = 2.0 - beta ** (-(pow_sbx + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (pow_sbx + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (pow_sbx + 1.0))
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        beta = 1.0 + (2.0 * (upp[i] - y2) / (y2 - y1))
        alpha = 2.0 - beta ** (-(pow_sbx + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (pow_sbx + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (pow_sbx + 1.0))
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        c1 = min(max(c1, low[i]), upp[i])
        c2 = min(max(c2, low[i]), upp[i])
        if rng.random() <= 0.5:
            child1[i], child2[i] = c2, c1
        else:
            child1[i], child2[i] = c1, c2

    problem.apply_bounds_x(child1)
    problem.apply_bounds_x(child2)

    return child1, child2


def binomial_crossover(x_target: np.ndarray, x_mutant: np.ndarray,
        cr: float, rng: np.random.Generator) -> np.ndarray:
    '''
    Binomial crossover operator.
    At least one dimension always comes from the mutant vector.
    
    Parameters:
    -----------
    x_target: np.ndarray
        Target individual.
    x_mutant: np.ndarray
        Mutant individual.
    cr: float
        Crossover rate.
    rng: np.random.Generator
        Random number generator.
    
    Returns:
    --------
    trial: np.ndarray
        Trial individual.
    '''
    n_dim = x_target.shape[0]
    trial = x_target.copy()
    j_rand = int(rng.integers(0, n_dim))
    rnd = rng.random(n_dim)
    mask = (rnd < cr) | (np.arange(n_dim) == j_rand)
    trial[mask] = x_mutant[mask]
    return trial


def polynomial_mutation(
        x: np.ndarray, problem: Problem,
        mut_rate: float = 1.0, pow_poly: float = 20.0,
        rng: np.random.Generator = None,
        ) -> np.ndarray:
    '''
    Polynomial mutation.
    '''
    if rng is None:
        rng = np.random.default_rng()

    out = x.copy()
    low = problem.data_settings.input_low
    upp = problem.data_settings.input_upp

    for i in range(problem.n_input):
        if rng.random() > mut_rate:
            continue

        span = upp[i] - low[i]
        if span <= 0.0:
            continue

        delta1 = (out[i] - low[i]) / span
        delta2 = (upp[i] - out[i]) / span
        rnd = rng.random()
        mut_pow = 1.0 / (pow_poly + 1.0)

        if rnd <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (pow_poly + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (pow_poly + 1.0))
            deltaq = 1.0 - val ** mut_pow

        out[i] += deltaq * span

    problem.apply_bounds_x(out)

    return out


def binary_tournament_selection(
        pool: Database, n_select: int,
        rng: np.random.Generator = None) -> List[Individual]:
    '''
    Binary tournament selection from a sorted population.

    Parameters:
    -----------
    pool: Database
        Population to select from.
    n_select: int
        Number of individuals to select.

    Returns:
    --------
    selected: List[Individual]
        Selected individuals.
    '''
    if pool.size <= 0:
        raise ValueError("Selection pool is empty.")

    if rng is None:
        rng = np.random.default_rng()

    selected: List[Individual] = []
    for _ in range(n_select):
        if pool.size > 1:
            pair = rng.choice(pool.size, size=2, replace=False)
            i, j = int(pair[0]), int(pair[1])
        else:
            i, j = 0, 0
        a = pool.individuals[i]
        b = pool.individuals[j]

        if a < b:
            selected.append(a)
        elif b < a:
            selected.append(b)
        else:
            selected.append(a if rng.random() < 0.5 else b)

    return selected


def perpendicular_distance(z: np.ndarray, direction_unit: np.ndarray) -> float:
    '''
    Perpendicular distance from objective vector z to a unit direction.
    '''
    t = float(np.dot(z, direction_unit))
    t = max(t, 0.0)
    return float(np.linalg.norm(z - t * direction_unit))


def reference_directions(ref_points: np.ndarray) -> np.ndarray:
    '''
    Normalize reference points to unit direction vectors.
    '''
    norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0e-12)
    return ref_points / norms


def associate_to_reference(z: np.ndarray, ref_dirs: np.ndarray) -> Tuple[int, float]:
    '''
    Associate one normalized objective vector to nearest reference direction.
    '''
    best_j, best_d = 0, float('inf')
    for j in range(ref_dirs.shape[0]):
        d = perpendicular_distance(z, ref_dirs[j, :])
        if d < best_d:
            best_d, best_j = d, j
    return best_j, best_d


def sample_de_rand_1_indices(rng: np.random.Generator, n_pop: int,
        i_target: int) -> Tuple[int, int, int]:
    '''
    Sample three parent indices for the DE/rand/1 mutation
    `v = x_r0 + F * (x_r1 - x_r2)`.
    This is a core function of the DE algorithm.

    Donors are drawn from indices in `0 .. n_pop - 1` other than
    `i_target`, so the target individual is excluded when possible. 
    If fewer than three distinct indices remain, 
    sampling is done with replacement. 
    If no candidate exists (`n_pop == 1`), returns `(0, 0, 0)`.

    Parameters:
    -----------
    rng: np.random.Generator
        NumPy random generator.
    n_pop: int
        Parent pool size.
    i_target: int
        Index of the trial's target individual in the parent pool.

    Returns:
    --------
    r0, r1, r2: Tuple[int, int, int]
        Donor indices for `x_r0`, `x_r1`, and `x_r2`.
    '''
    allowed = [j for j in range(n_pop) if j != i_target]
    if not allowed:
        return 0, 0, 0
    replace = len(allowed) < 3
    triplet = rng.choice(allowed, size=3, replace=replace)
    return int(triplet[0]), int(triplet[1]), int(triplet[2])

