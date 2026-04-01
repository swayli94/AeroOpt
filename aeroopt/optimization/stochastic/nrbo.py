'''
NRBO: Newton-Raphson-based Optimizer

NRBO is a population-based metaheuristic algorithm for single-objective continuous optimization problems.
NRBO is inspired by the Newton-Raphson method used in numerical analysis for finding roots of real-valued functions.
The algorithm combines the exploration capabilities of population-based methods with the exploitation power of gradient-based approaches.

Key Features:

- Population-based approach: Maintains a population of solutions to explore the search space
- Newton-Raphson Search Rule (NRSR): Utilizes a modified Newton-Raphson update rule for generating new solutions
- Trap Avoidance Operator (TAO): Includes a mechanism to escape local optima
- Dynamic adaptation: The algorithm adapts its search behavior based on the current iteration

References:

    R. Sowmya, M. Premkumar, and P. Jangir. Newton-raphson-based optimizer:
    a new population-based metaheuristic algorithm for continuous optimization problems.
    Engineering Applications of Artificial Intelligence, 128:107532, 2024. doi:10.1016/j.engappai.2023.107532.

    https://pymoo.org/algorithms/soo/nrbo.html#nb-nrbo

    https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/soo/nonconvex/nrbo.py

'''

from __future__ import annotations

from typing import Callable

import numpy as np

from aeroopt.core import (
    Problem,
    Individual,
    Database,
    MultiProcessEvaluation,
)
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.moea import Algorithm, DominanceBasedAlgorithm
from aeroopt.optimization.settings import SettingsNRBO, SettingsOptimization


class NRBO(Algorithm):
    '''
    NRBO operators (NRSR + TAO) adapted to the aeroopt database workflow.
    '''

    @staticmethod
    def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
        '''
        Perform element-wise division with epsilon protection for near-zero denominators.

        Parameters:
        -----------
        numer: np.ndarray
            Numerator array.
        denom: np.ndarray
            Denominator array.

        Returns:
        --------
        result: np.ndarray
            Element-wise quotient with numerically safe denominator handling.
        '''
        eps = 1.0e-12
        d = np.asarray(denom, dtype=float).copy()
        mask = np.abs(d) < eps
        d[mask] = np.sign(d[mask]) * eps
        d[d == 0.0] = eps
        return np.asarray(numer, dtype=float) / d

    @staticmethod
    def _search_rule(
            x_best: np.ndarray,
            x_worst: np.ndarray,
            x_now: np.ndarray,
            rho: np.ndarray,
            rng: np.random.Generator,
            ) -> tuple:
        '''
        Apply the Newton-Raphson Search Rule (NRSR) to produce two trial anchors.

        Parameters:
        -----------
        x_best: np.ndarray
            Best decision vector in the current parent pool.
        x_worst: np.ndarray
            Worst decision vector in the current parent pool.
        x_now: np.ndarray
            Current individual's decision vector.
        rho: np.ndarray
            Differential perturbation vector used by NRBO.
        rng: np.random.Generator
            Random number generator.

        Returns:
        --------
        result: tuple
            Tuple `(x1, x2)` where both are candidate anchor vectors used to
            construct the final offspring.
        '''
        dim = x_now.shape[0]

        dx = rng.random(dim) * np.abs(x_best - x_now)
        den = x_worst + x_best - 2.0 * x_now
        nrsr = rng.standard_normal() * NRBO._safe_divide(
            (x_worst - x_best) * dx, 2.0 * den)
        z = x_now - nrsr

        r1 = float(rng.random())
        z_mean = float(np.mean(z + x_now))
        y_w = r1 * (z_mean + r1 * dx)
        y_b = r1 * (z_mean - r1 * dx)
        den2 = 2.0 * (y_w + y_b - 2.0 * x_now)
        nrsr_2 = rng.standard_normal() * NRBO._safe_divide((y_w - y_b) * dx, den2)

        step = nrsr_2 - rho
        x1 = x_now - step
        x2 = x_best - step
        return x1, x2

    @staticmethod
    def _best_worst_indices_by_objective(
            db: Database,
            ) -> tuple:
        '''
        Find the best and worst individual indices for single-objective fitness.

        Parameters:
        -----------
        db: Database
            Database whose individuals already contain evaluated objectives.

        Returns:
        --------
        result: tuple
            Tuple `(i_best, i_worst)` as local indices into `db.individuals`.
        '''
        if db.size <= 0:
            raise ValueError('Cannot select best/worst from an empty database.')
        scaled_ys = db.get_unified_objectives(scale=True)
        if scaled_ys.shape[1] != 1:
            raise ValueError('NRBO only supports single-objective problems.')
        vals = scaled_ys[:, 0]
        i_best = int(np.argmin(vals))
        i_worst = int(np.argmax(vals))
        return i_best, i_worst

    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            max_iterations: int = 20,
            deciding_factor: float = 0.6,
            rng: np.random.Generator | None = None,
            ) -> None:
        '''
        Generate one NRBO offspring per parent and write them into `db_candidate`.

        Parameters:
        -----------
        db: Database
            Population database.
        db_candidate: Database
            Output database to be overwritten by new candidates.
        population_size: int
            Target parent-pool/offspring count.
        iteration: int
            Current optimization iteration index.
        max_iterations: int
            Maximum number of optimization iterations.
        deciding_factor: float
            Probability of applying the TAO operator.
        rng: np.random.Generator
            Optional NumPy random generator.
        '''
        if db.size <= 0:
            raise RuntimeError('No individuals available for NRBO evolution.')
        if db.problem.n_objective != 1:
            raise ValueError('NRBO supports only single-objective problems.')

        if rng is None:
            rng = np.random.default_rng()

        parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            db, population_size)
        n_pop = parents.size
        if n_pop <= 0:
            raise RuntimeError('Empty parent pool while generating NRBO candidates.')

        X = np.asarray([indi.x for indi in parents.individuals], dtype=float)
        i_best, i_worst = NRBO._best_worst_indices_by_objective(parents)
        x_best = X[i_best].copy()
        x_worst = X[i_worst].copy()

        # Keep the same decay structure as the paper/pymoo implementation.
        delta = (1.0 - (2.0 * float(iteration)) / max(float(max_iterations), 1.0)) ** 5

        db_candidate.empty_database()
        problem = db_candidate.problem

        for i in range(n_pop):
            if n_pop >= 3:
                idx = np.arange(n_pop, dtype=int)
                idx = idx[idx != i]
                r1, r2 = rng.choice(idx, size=2, replace=False)
                x_r1 = X[int(r1)]
                x_r2 = X[int(r2)]
            else:
                x_r1 = x_best
                x_r2 = x_worst

            a, b = rng.random(2)
            rho = a * (x_best - X[i]) + b * (x_r1 - x_r2)
            x1, x2 = NRBO._search_rule(
                x_best=x_best, x_worst=x_worst, x_now=X[i], rho=rho, rng=rng)

            x3 = X[i] - delta * (x2 - x1)
            r = float(rng.random())
            x_new : np.ndarray = r * (r * x1 + (1.0 - r) * x2) + (1.0 - r) * x3

            # Apply the Trap Avoidance Operator (TAO)
            # Adding a random perturbation to avoid getting trapped in a local optimum
            if float(rng.random()) < float(deciding_factor):
                theta1 = float(rng.uniform(-1.0, 1.0))
                theta2 = float(rng.uniform(-0.5, 0.5))
                beta = 0.0 if float(rng.random()) > 0.5 else 1.0
                u1 = beta * 3.0 * float(rng.random()) + (1.0 - beta)
                u2 = beta * float(rng.random()) + (1.0 - beta)
                tmp : np.ndarray = (
                    theta1 * (u1 * x_best - u2 * X[i])
                    + theta2 * delta * (u1 * float(np.mean(X[i])) - u2 * X[i])
                )
                x_new = x_new + tmp if u1 < 0.5 else x_best + tmp

            problem.apply_bounds_x(x_new)
            indi = Individual(problem=problem, x=x_new)
            indi.source = 'evolutionary_operator'
            indi.generation = int(iteration)
            db_candidate.add_individual(
                indi,
                check_duplication=True,
                check_bounds=True,
                deepcopy=False,
                print_warning_info=False,
            )


class OptNRBO(OptBaseFramework):
    '''
    Optimization driver using NRBO for candidate generation.
    
    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    algorithm_settings: SettingsNRBO
        NRBO-specific settings.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    rng: np.random.Generator
        Optional NumPy random generator.
    '''

    def __init__(
            self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsNRBO,
            user_func: Callable|None = None,
            mp_evaluation: MultiProcessEvaluation|None = None,
            user_func_supports_parallel: bool = False,
            rng: np.random.Generator|None = None,
            save_result_files: bool = True,
            logging: bool = True,
            ):
        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
        )
        self.algorithm_settings = algorithm_settings
        self.rng = rng

        if self.problem.n_objective != 1:
            raise ValueError('OptNRBO only supports single-objective problems.')

    def generate_candidate_individuals(self) -> None:
        '''
        Generate NRBO candidates for the current iteration.
        '''
        if self.db_valid.size <= 0:
            _db = self.db_total
        else:
            _db = self.db_valid

        NRBO.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            deciding_factor=self.algorithm_settings.deciding_factor,
            rng=self.rng,
        )

    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid archive.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)
