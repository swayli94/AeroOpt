'''
Differential evolution (DE/rand/1/bin).

Parents each iteration are a truncated pool of size `population_size` from
`db` (same environmental selection as NSGA-II: ranks + crowding),
so the operator fits the archive-based workflow of `OptBaseFramework`.
'''

from __future__ import annotations

from typing import Callable

import numpy as np

from aeroopt.core import (
    Individual,
    Database,
    MultiProcessEvaluation,
    Problem,
)
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.moea import Algorithm, DominanceBasedAlgorithm
from aeroopt.optimization.settings import SettingsDE, SettingsOptimization
from aeroopt.optimization.utils import sample_de_rand_1_indices, binomial_crossover


class DiffEvolution(Algorithm):
    '''
    Classic differential evolution (DE/rand/1/bin) with binomial crossover.
    '''

    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            scale_factor: float = 0.5,
            cross_rate: float = 0.8,
            rng: np.random.Generator|None = None,
            ) -> None:
        '''
        Build a parent pool from `db` (truncation with dominance-based algorithm),
        then apply binomial crossover and mutation to generate candidate individuals.
        '''
        if db.size <= 0:
            raise RuntimeError(
                'No individuals available for differential evolution.')

        if rng is None:
            rng = np.random.default_rng()

        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            db, population_size)
        n_pop = temp_parents.size
        problem = db_candidate.problem

        db_candidate.empty_database()

        for i in range(population_size):
            i_t = i % n_pop
            x_t = temp_parents.individuals[i_t].x
            r0, r1, r2 = sample_de_rand_1_indices(rng, n_pop, i_t)
            x0 = temp_parents.individuals[r0].x
            x1 = temp_parents.individuals[r1].x
            x2 = temp_parents.individuals[r2].x
            mutant = x0 + scale_factor * (x1 - x2)
            trial_x = binomial_crossover(
                x_t, mutant, cross_rate, rng)
            problem.apply_bounds_x(trial_x)

            indi = Individual(problem=problem, x=trial_x)
            indi.source = 'evolutionary_operator'
            indi.generation = iteration
            db_candidate.add_individual(
                indi,
                check_duplication=True,
                check_bounds=True,
                deepcopy=False,
                print_warning_info=False,
            )


class OptDE(OptBaseFramework):
    '''
    Optimization driver using differential evolution for offspring generation.
    
    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    algorithm_settings: SettingsDE
        Differential evolution-specific settings.
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
            algorithm_settings: SettingsDE,
            user_func: Callable|None = None,
            mp_evaluation: MultiProcessEvaluation|None = None,
            user_func_supports_parallel: bool = False,
            rng: np.random.Generator|None = None,
            logging: bool = True,
            ):
        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            logging=logging,
        )
        self.algorithm_settings = algorithm_settings
        self.rng = rng

    def generate_candidate_individuals(self) -> None:
        '''
        Generate candidate individuals from the total or valid database.
        '''
        if self.db_valid.size <= 0:
            _db = self.db_total
        else:
            _db = self.db_valid
        
        DiffEvolution.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            scale_factor=self.algorithm_settings.scale_factor,
            cross_rate=self.algorithm_settings.cross_rate,
            rng=self.rng,
        )
        
    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)
