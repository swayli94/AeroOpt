'''
NSGA-II implementation.
'''

from typing import Callable

import numpy as np

from aeroopt.core import (
    Problem, Individual, Database,
    MultiProcessEvaluation
)
from aeroopt.optimization.moea import Algorithm, DominanceBasedAlgorithm
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.settings import SettingsOptimization, SettingsNSGAII
from aeroopt.optimization.utils import (
    binary_tournament_selection,
    polynomial_mutation,
    sbx_crossover,
)


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
        if db.size <= 0:
            raise RuntimeError(
                "No individuals available for NSGA-II evolution.")

        if rng is None:
            rng = np.random.default_rng()

        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            db, population_size)
        mating_population = binary_tournament_selection(
            pool=temp_parents, n_select=population_size, rng=rng)

        db_candidate.empty_database()
        n_pairs = int(np.ceil(population_size / 2))

        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = sbx_crossover(
                p1.x, p2.x, problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx, rng=rng)

            x1 = polynomial_mutation(
                x1, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly, rng=rng)
            x2 = polynomial_mutation(
                x2, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly, rng=rng)

            for x_child in (x1, x2):
                if db_candidate.size >= population_size:
                    break
                indi = Individual(problem=db_candidate.problem, x=x_child)
                indi.source = 'evolutionary_operator'
                indi.generation = iteration
                db_candidate.add_individual(
                    indi, check_duplication=True, check_bounds=True,
                    deepcopy=False, print_warning_info=False)


class OptNSGAII(OptBaseFramework):
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
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
        )
    
        self.algorithm_settings = algorithm_settings
        self.rng = rng
    
    #* Main procedures
    
    def generate_candidate_individuals(self) -> None:
        '''
        Generate candidate individuals from the population database.

        A temporary parent database (size `population_size`) is built from
        `db` via `DominanceBasedAlgorithm.build_temporary_parent_database`; 
        tournament selection and variation use only that pool. 
        `db` can be a valid archive or total pool.
        '''
        mut_rate = self.algorithm_settings.mut_rate / max(self.problem.n_input, 1)
        
        if self.db_valid.size <= 0:
            _db = self.db_total
        else:
            _db = self.db_valid
        
        NSGAII.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=mut_rate,
            pow_poly=self.algorithm_settings.pow_poly,
            rng=self.rng,
        )
 
    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)

