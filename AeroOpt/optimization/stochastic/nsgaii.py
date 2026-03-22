'''
NSGA-II implementation.
'''

import random
from typing import List, Tuple, Callable

import numpy as np

from AeroOpt.core import (
    Problem, Individual, Database,
    MultiProcessEvaluation
)
from AeroOpt.optimization.stochastic.base import (
    OptEvolutionaryFramework, EvolutionaryAlgorithm
)
from AeroOpt.optimization.settings import SettingsOptimization, SettingsNSGAII


class NSGAII(EvolutionaryAlgorithm):
    '''
    NSGA-II evolutionary algorithm.
    
    Parameters:
    -----------
    settings_name: str
        Name of the `SettingsNSGAII` dictionary in the settings file.
    fname_settings: str
        Name of the settings file.
    '''
    def __init__(self, settings_name: str = "default",
            fname_settings: str = 'settings.json'):
        
        super().__init__(algorithm_name='NSGAII')
        
        self.settings = SettingsNSGAII(
            name=settings_name, fname_settings=fname_settings)

    @staticmethod
    def sbx_crossover(x1: np.ndarray, x2: np.ndarray, problem: Problem,
        cross_rate: float = 1.0, pow_sbx: float = 20.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Simulated Binary Crossover (SBX) Operation.

        Deb, Kalyanmoy, and Ram Bhushan Agrawal. 
        "Simulated binary crossover for continuous search space."
        Complex systems 9.2 (1995): 115-148.
        '''
        child1 = x1.copy()
        child2 = x2.copy()
        if random.random() > cross_rate:
            return child1, child2

        low = problem.data_settings.input_low
        upp = problem.data_settings.input_upp
        precision = problem.data_settings.input_precision

        for i in range(problem.n_input):
            if random.random() > 0.5:
                continue
            if abs(x1[i] - x2[i]) <= precision[i]:
                continue

            y1, y2 = (x1[i], x2[i]) if x1[i] < x2[i] else (x2[i], x1[i])
            rand = random.random()

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
            if random.random() <= 0.5:
                child1[i], child2[i] = c2, c1
            else:
                child1[i], child2[i] = c1, c2

        problem.apply_bounds_x(child1)
        problem.apply_bounds_x(child2)
        
        return child1, child2

    @staticmethod
    def polynomial_mutation(x: np.ndarray, problem: Problem,
        mut_rate: float = 1.0, pow_poly: float = 20.0,
        ) -> np.ndarray:
        '''
        Polynomial mutation.
        '''
        out = x.copy()
        low = problem.data_settings.input_low
        upp = problem.data_settings.input_upp

        for i in range(problem.n_input):
            if random.random() > mut_rate:
                continue

            span = upp[i] - low[i]
            if span <= 0.0:
                continue

            delta1 = (out[i] - low[i]) / span
            delta2 = (upp[i] - out[i]) / span
            rnd = random.random()
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

    @staticmethod
    def binary_tournament_selection(pool: Database,
                            n_select: int) -> List[Individual]:
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

        selected: List[Individual] = []
        for _ in range(n_select):
            i, j = random.sample(range(pool.size), 2) if pool.size > 1 else (0, 0)
            a = pool.individuals[i]
            b = pool.individuals[j]

            if a < b:
                selected.append(a)
            elif b < a:
                selected.append(b)
            else:
                selected.append(a if random.random() < 0.5 else b)
            
        return selected

    @staticmethod
    def build_temporary_parent_database(
            db_valid: Database, population_size: int) -> Database:
        '''
        Build a temporary parent pool of size at most `population_size` from the
        valid archive (deep copy), using NSGA-II environmental selection
        (ranks + crowding distance). Does not modify `db_valid`.
        '''
        if db_valid.size <= 0:
            raise ValueError("Cannot build parent database from an empty valid archive.")

        db_work = db_valid.get_sub_database(
            index_list=list(range(db_valid.size)), deepcopy=True)
        fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(
            db_work, is_valid_database=True)
        EvolutionaryAlgorithm.assign_crowding_distance(db_work, fronts)
        if db_work.size <= population_size:
            return db_work
        idx = EvolutionaryAlgorithm.select_population_indices(
            db_work, fronts, population_size)
        return db_work.get_sub_database(index_list=idx, deepcopy=True)

    @staticmethod
    def generate_candidate_individuals(
            db_valid: Database, db_candidate: Database,
            population_size: int, iteration: int,
            cross_rate: float = 1.0, pow_sbx: float = 20.0,
            mut_rate: float = 1.0, pow_poly: float = 20.0) -> None:
        '''
        Generate candidate individuals from the valid database.

        A temporary parent database (size `population_size`) is built from
        `db_valid` via `build_temporary_parent_database`; tournament selection
        and variation use only that pool. `db_valid` is not reinterpreted as
        the parent generation.
        
        Parameters:
        -----------
        db_valid: Database
            Valid database.
        db_candidate: Database
            Candidate database.
        population_size: int
            Population size.
        iteration: int
            Iteration number.
        cross_rate: float
            Crossover rate.
        pow_sbx: float
            Simulated binary crossover power.
        mut_rate: float
            Mutation rate.
        pow_poly: float
            Polynomial mutation power.
        
        Returns:
        --------
        db_candidate: Database
            Candidate database.
        '''
        if db_valid.size <= 0:
            raise RuntimeError("No valid individuals available for NSGA-II evolution.")

        temp_parents = NSGAII.build_temporary_parent_database(
            db_valid, population_size)
        mating_population = NSGAII.binary_tournament_selection(
            pool=temp_parents, n_select=population_size)

        db_candidate.empty_database()

        n_pairs = int(np.ceil(population_size / 2))
        
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = NSGAII.sbx_crossover(p1.x, p2.x, problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx)

            x1 = NSGAII.polynomial_mutation(x1, problem=db_candidate.problem,
                                            mut_rate=mut_rate, pow_poly=pow_poly)

            x2 = NSGAII.polynomial_mutation(x2, problem=db_candidate.problem,
                                            mut_rate=mut_rate, pow_poly=pow_poly)

            children = [x1, x2]
            for k, x_child in enumerate(children):
                if db_candidate.size >= population_size:
                    break
                indi = Individual(problem=db_candidate.problem, x=x_child)
                indi.source = "GA"
                indi.generation = iteration
                db_candidate.add_individual(indi, check_duplication=True,
                            check_bounds=True, deepcopy=False, print_warning_info=False)


class OptNSGAII(OptEvolutionaryFramework):
    '''
    NSGA-II optimization.
    '''
    def __init__(self,
        problem: Problem,
        optimization_settings: SettingsOptimization,
        evolutionary_algorithm: NSGAII,
        user_func: Callable = None,
        mp_evaluation: MultiProcessEvaluation = None
        ):
        
        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            evolutionary_algorithm=evolutionary_algorithm,
            user_func=user_func,
            mp_evaluation=mp_evaluation)
    
    #* Main procedures
    
    def generate_candidate_individuals(self) -> None:
        '''
        Generate offspring using a temporary parent pool truncated from `db_valid`.
        '''
        mute_rate = self.evolutionary_algorithm.settings.mut_rate / max(self.problem.n_input, 1)
        
        NSGAII.generate_candidate_individuals(
            db_valid=self.db_valid,
            db_candidate=self.db_candidate,
            population_size=self.population_size, 
            iteration=self.iteration,
            cross_rate=self.evolutionary_algorithm.settings.cross_rate,
            pow_sbx=self.evolutionary_algorithm.settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.evolutionary_algorithm.settings.pow_poly)
