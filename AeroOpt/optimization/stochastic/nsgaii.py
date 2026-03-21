'''
NSGA-II implementation.
'''

import random
from typing import List, Tuple, Callable

import numpy as np

from AeroOpt.core import (
    Database, Problem, SettingsOptimization, MultiProcessEvaluation
)
from AeroOpt.core.individual import Individual
from AeroOpt.core.settings import SettingsNSGAII
from AeroOpt.optimization.base import PostProcess, PreProcess
from AeroOpt.optimization.stochastic.base import (
    OptEvolutionaryFramework, EvolutionaryAlgorithm
)


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


class OptNSGAII(OptEvolutionaryFramework):
    '''
    NSGA-II optimization.
    '''
    def __init__(self,
        problem: Problem,
        optimization_settings: SettingsOptimization,
        evolutionary_algorithm: NSGAII,
        user_func: Callable = None,
        mp_evaluation: MultiProcessEvaluation = None,
        pre_process: PreProcess = None,
        post_process: PostProcess = None,
        ):
        
        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            evolutionary_algorithm=evolutionary_algorithm,
            user_func=user_func,
            mp_evaluation=mp_evaluation,
            pre_process=pre_process,
            post_process=post_process)
    
    @property
    def cross_rate(self) -> float:
        '''
        Crossover rate.
        '''
        return self.evolutionary_algorithm.settings.cross_rate
    
    @property
    def mute_rate(self) -> float:
        '''
        Mutation rate scaled by the number of input variables.
        '''
        return self.evolutionary_algorithm.settings.mut_rate / max(self.problem.n_input, 1)
    
    @property
    def pow_poly(self) -> float:
        '''
        Polynomial mutation power.
        '''
        return self.evolutionary_algorithm.settings.pow_poly

    @property
    def pow_sbx(self) -> float:
        '''
        Simulated binary crossover power.
        '''
        return self.evolutionary_algorithm.settings.pow_sbx
        
    #* Main procedures
        
    def initialize_population(self) -> None:
        '''
        Initialize population if no resumed database exists.
        '''
        self.iteration = 1

        if self.db_total.size <= 0:
            
            xs = np.random.rand(self.population_size, self.problem.n_input)
            xs = self.problem.scale_x(xs, reverse=True)
            
            self.db_candidate = Database(self.problem, database_type="population")
            for x in xs:
                indi = Individual(self.problem, x=np.array(x, dtype=float))
                indi.source = "random"
                indi.generation = self.iteration
                self.db_candidate.add_individual(
                    indi,
                    check_duplication=False,
                    check_bounds=True,
                    deepcopy=False,
                )

            if self.pre_process is not None:
                self.pre_process.apply()
            self.evaluate_db_candidate()
            if self.post_process is not None:
                self.post_process.apply()

        self.select_valid_elite_from_total()
        self.log(f"Initial population prepared: valid={self.db_valid.size}", level=1)

    def generate_candidate_individuals(self) -> None:
        '''
        Generate offspring from current valid population.
        '''
        if self.db_valid.size <= 0:
            raise RuntimeError("No valid individuals available for NSGA-II evolution.")

        mating_population = NSGAII.binary_tournament_selection(
            pool=self.db_valid, n_select=self.population_size)
        self.db_candidate = Database(self.problem, database_type="population")

        n_pairs = int(np.ceil(self.population_size / 2))
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, self.population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = NSGAII.sbx_crossover(
                p1.x,
                p2.x,
                self.problem,
                cross_rate=self.cross_rate,
                pow_sbx=self.pow_sbx,
            )

            x1 = NSGAII.polynomial_mutation(
                x1,
                self.problem,
                mut_rate=self.mute_rate,
                pow_poly=self.pow_poly,
            )
            x2 = NSGAII.polynomial_mutation(
                x2,
                self.problem,
                mut_rate=self.mute_rate,
                pow_poly=self.pow_poly,
            )

            children = [x1, x2]
            for k, x_child in enumerate(children):
                if self.db_candidate.size >= self.population_size:
                    break
                indi = Individual(self.problem, x=np.array(x_child, dtype=float))
                indi.source = "GA"
                indi.generation = self.iteration
                self.db_candidate.add_individual(
                    indi,
                    check_duplication=True,
                    check_bounds=True,
                    deepcopy=False,
                )

