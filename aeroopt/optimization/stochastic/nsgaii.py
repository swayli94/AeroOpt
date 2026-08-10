'''
NSGA-II implementation.
'''

from __future__ import annotations

from typing import Callable

import numpy as np

from aeroopt.core import (
    Problem, Database,
    MultiProcessEvaluation
)
from aeroopt.optimization.moea import Algorithm, DominanceBasedAlgorithm
from aeroopt.optimization.base import OptGeneticFramework
from aeroopt.optimization.settings import SettingsOptimization, SettingsNSGAII
from aeroopt.optimization.utils import fill_candidates_by_sbx_and_mutation


class NSGAII(Algorithm):
    '''
    NSGA-II operators.
    '''
    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            cross_rate: float = 1.0,
            pow_sbx: float = 20.0,
            mut_rate: float = 1.0,
            pow_poly: float = 20.0,
            rng: np.random.Generator|None = None,
            ) -> None:
        '''
        Build a parent pool from `db` by rank-and-crowding truncation, then fill
        `db_candidate` with SBX + polynomial-mutation offspring.
        '''
        if db.size <= 0:
            raise RuntimeError(
                "No individuals available for NSGA-II evolution.")

        if rng is None:
            rng = np.random.default_rng()

        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            db, population_size)

        fill_candidates_by_sbx_and_mutation(
            parents=temp_parents,
            db_candidate=db_candidate,
            population_size=population_size,
            iteration=iteration,
            cross_rate=cross_rate, pow_sbx=pow_sbx,
            mut_rate=mut_rate, pow_poly=pow_poly,
            rng=rng)


class OptNSGAII(OptGeneticFramework):
    '''
    NSGA-II optimization.

    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    algorithm_settings: SettingsNSGAII
        NSGA-II-specific settings.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    rng: np.random.Generator
        Optional NumPy random generator.
    '''
    def __init__(self,
        problem: Problem,
        optimization_settings: SettingsOptimization,
        algorithm_settings: SettingsNSGAII,
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
            algorithm_settings=algorithm_settings,
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
            rng=rng,
        )

    #* Main procedures

    def generate_candidate_individuals(self) -> None:
        '''
        Generate candidate individuals from the population database.

        A temporary parent database (size `population_size`) is built via
        `DominanceBasedAlgorithm.build_temporary_parent_database`; tournament
        selection and variation use only that pool.
        '''
        NSGAII.generate_candidate_individuals(
            db=self.select_population_database(),
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=self.mut_rate_per_variable,
            pow_poly=self.algorithm_settings.pow_poly,
            rng=self.rng,
        )

