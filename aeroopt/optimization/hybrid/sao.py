'''
Surrogate-assisted Optimization (SAO)
'''
from typing import List, Callable, Tuple

import numpy as np
import functools
import copy

from aeroopt.core import (
    Problem, Individual, 
    MultiProcessEvaluation
)
from aeroopt.optimization.settings import SettingsOptimization, SettingsDE
from aeroopt.optimization.base import OptBaseFramework, PreProcess, PostProcess
from aeroopt.utils.surrogate import SurrogateModel
from aeroopt.optimization.moea import DominanceBasedAlgorithm
from aeroopt.optimization.stochastic.de import DiffEvolution


def _surrogate_user_func(
    xs: np.ndarray,
    surrogate: SurrogateModel,
    **kwargs,
    ) -> Tuple[List[bool], np.ndarray]:
    ys = surrogate.predict_for_adaptive_sampling(np.asarray(xs, dtype=float), **kwargs)
    return [True] * len(xs), np.asarray(ys, dtype=float)


class PostProcessSAO(PostProcess):
    '''
    Post-processing of the `db_total` database after evaluation.
    And evaluate the performance of the surrogate model by comparing the prediction
    and actual values of the individuals in the `db_candidate` database.
    
    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object.
    '''
    def __init__(self, opt: OptBaseFramework, surrogate: SurrogateModel):

        super().__init__(opt)

        self.surrogate = surrogate
        
    def apply(self) -> None:
        '''
        Apply the post-processing to the `db_total` database.
        
        - Find individuals in `db_candidate` from 'evolutionary_operator'
          and 'surrogate_prediction' sources, denoted as 'E' and 'S' respectively.
        - Evaluate the performance of the surrogate model by
          comparing the prediction and actual values of the individuals 'S' and 'E'.
        - Evaluate the contribution of the surrogate model by
          comparing the dominance of the individuals 'S' and 'E'.
        '''
        if self.surrogate.size <= 0 or self.opt.db_candidate.size <= 0:
            self.opt.log(
                'Surrogate performance: skipped (model not trained yet or no candidates).',
                level=2, prefix='    ')
            return None

        self.opt.log(f'Evaluating the performance and contribution of the surrogate model.', level=1)

        xs = self.opt.db_candidate.get_xs(scale=False)
        ys_actual = self.opt.db_candidate.get_ys(scale=False)
        ys_actual = ys_actual[:, self.opt.index_outputs_for_surrogate]
        
        index_E = []
        index_S = []
        for i, indi in enumerate(self.opt.db_candidate.individuals):
            if indi.source == 'evolutionary_operator':
                index_E.append(i)
            elif indi.source == 'surrogate_prediction':
                index_S.append(i)
        index_E = np.array(index_E)
        index_S = np.array(index_S)
        
        self.opt.log(f'Number of "S" individuals (from surrogate predictions):  {len(index_S)}', level=2, prefix='    ')
        self.opt.log(f'Number of "E" individuals (from evolutionary operators): {len(index_E)}', level=2, prefix='    ')

        performance_dict_S = self.surrogate.evaluate_performance(xs[index_S], ys_actual[index_S])
        performance_dict_E = self.surrogate.evaluate_performance(xs[index_E], ys_actual[index_E])
        self.opt.log(f'Surrogate performance on "S" individuals:', level=2, prefix='    ')
        for key, value in performance_dict_S.items():
            self.opt.log(f'{key:10s}: {value}', level=2, prefix='    ')
        self.opt.log(f'Surrogate performance on "E" individuals:', level=2, prefix='    ')
        for key, value in performance_dict_E.items():
            self.opt.log(f'{key:10s}: {value}', level=2, prefix='    ')

        # Check dominance within db_candidate
        temp_db = copy.deepcopy(self.opt.db_candidate)
        temp_db.eliminate_invalid_individuals()
        index_fronts = DominanceBasedAlgorithm.non_dominated_ranking(temp_db)
        n_E = 0
        n_S = 0
        for ii in index_fronts[0]:
            if temp_db.individuals[ii].source == 'evolutionary_operator':
                n_E += 1
            elif temp_db.individuals[ii].source == 'surrogate_prediction':
                n_S += 1
        self.opt.log(f'Sources of Pareto front of the candidates:', level=2, prefix='    ')
        self.opt.log(f'Number of "S" individuals: {n_S}', level=2, prefix='    ')
        self.opt.log(f'Number of "E" individuals: {n_E}', level=2, prefix='    ')
        

class SAO(OptBaseFramework):
    '''
    Surrogate-assisted Optimization (SAO).
    
    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    algorithm_settings: SettingsDE
        Settings of the differential evolution (DE) algorithm in the main optimization loop.
    surrogate: SurrogateModel
        Surrogate model for optimization.
    opt_on_surrogate: OptBaseFramework
        Optimization object on the surrogate model.
    ratio_from_surrogate: float
        Ratio of the number of candidate individuals from the surrogate model.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    pre_process: PreProcess
        Pre-processing of the `db_candidate` database that are predicted by the surrogate model.
    post_process: PostProcess
        Post-processing of the `db_total` and `db_valid` databases.
        Also evaluate the performance of the surrogate model by
        comparing the prediction and actual values of the candidates.
    rng: np.random.Generator
        Random generator for differential evolution in the main loop.
        If None, ``numpy.random.default_rng()`` is used.
        
    Attributes:
    -----------
    outputs_for_surrogate: List[str]
        List of names of outputs that are predicted by the surrogate model.
    index_outputs_for_surrogate: np.ndarray
        Indices of the output variables that are predicted by the surrogate model.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsDE,
            surrogate: SurrogateModel,
            opt_on_surrogate: OptBaseFramework,
            ratio_from_surrogate: float = 0.5,
            user_func: Callable = None,
            mp_evaluation: MultiProcessEvaluation = None,
            pre_process: PreProcess = None,
            post_process: PostProcessSAO = None,
            rng: np.random.Generator = None):
        
        super().__init__(
            problem,
            optimization_settings,
            user_func,
            mp_evaluation=mp_evaluation,
        )
        
        self.surrogate = surrogate
        self.opt_on_surrogate = opt_on_surrogate
        
        self.pre_process = pre_process
        self.post_process = post_process
        
        self.algorithm_settings = algorithm_settings
        self.ratio_from_surrogate = ratio_from_surrogate
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Outputs for surrogate model
        self._set_outputs_for_surrogate()
        
    @property
    def outputs_for_surrogate(self) -> List[str]:
        '''
        List of names of outputs that are predicted by the surrogate model.
        '''
        return self._outputs_for_surrogate
    
    @property
    def index_outputs_for_surrogate(self) -> np.ndarray:
        '''
        Indices of the output variables that are predicted by the surrogate model.
        
        This is used to select the output variables from the original output variables.
        '''
        return self._index_outputs_for_surrogate
    
    def _set_outputs_for_surrogate(self) -> None:
        '''
        Set the output variables that are predicted by the surrogate model.
        '''
        self._outputs_for_surrogate = self.surrogate.problem.name_output
        
        try:
            self._index_outputs_for_surrogate = np.array(
                [self.problem.name_output.index(name) 
                for name in self._outputs_for_surrogate], dtype=int)
            
        except Exception as e:
            self.log(f'Outputs defined in the surrogate model are not in the global optimization problem.',
                        level=2, prefix='    ')
            self.log(f'Error message: {e}', level=2, prefix='    ')
            raise Exception('Surrogate model problem error.') from e
    
    #TODO: Can be adapted
    def update_parameters(self) -> None:
        '''
        Surrogate model training with `db_valid` database.
        '''
        xs = self.db_valid.get_xs(scale=False)
        ys = self.db_valid.get_ys(scale=False)
        ys = ys[:, self.index_outputs_for_surrogate]
        self.surrogate.train(xs, ys)
    
    def _generate_candidate_individuals_from_evolutionary_operators(self) -> None:
        '''
        Generate candidate individuals using the evolutionary operators.
        '''
        DiffEvolution.generate_candidate_individuals(
            db_valid=self.db_valid,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            scale_factor=self.algorithm_settings.scale_factor,
            cross_rate=self.algorithm_settings.cross_rate,
            rng=self.rng,
        )
    
    def _generate_candidate_individuals_from_surrogate(self, n_candidates: int) -> None:
        '''
        Generate candidate individuals using the surrogate model.
        
        Do not empty the `db_candidate` database.
        If the new candidates are not duplicated with the existing candidates,
        replace the last existing candidates with the new candidates.
        '''
        index_of_candidate_to_replace = self.db_candidate.size - 1
        
        self.opt_on_surrogate.initialize()

        # Set the user function for the surrogate model
        user_func = functools.partial(_surrogate_user_func, surrogate=self.surrogate)
        self.opt_on_surrogate.user_func = user_func
        self.opt_on_surrogate.user_func_supports_parallel = True
        
        # Optimization on the surrogate model
        self.opt_on_surrogate.main()
        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            self.opt_on_surrogate.db_valid, n_candidates)
        n_pop = temp_parents.size
        
        # Get the candidate individuals from the surrogate model
        for i in range(n_pop):

            x = temp_parents.individuals[i].x

            indi = Individual(problem=self.problem, x=x)
            indi.source = 'surrogate_prediction'
            indi.generation = self.iteration
            added, warning_text = self.db_candidate.add_individual(
                indi,
                check_duplication=True,
                check_bounds=True,
                deepcopy=False,
                print_warning_info=False,
            )
            
            if added:
                # Store the prediction of the surrogate model
                y_predicted = np.zeros(self.problem.n_output)
                y_predicted[self.index_outputs_for_surrogate] = temp_parents.individuals[i].y
                indi._y_predicted = y_predicted
                
                # Keep the size of db_candidate equal to population_size
                # By deleting the last candidate from original db_candidate
                if self.db_candidate.size > self.population_size:
                    self.db_candidate.delete_individual(index=index_of_candidate_to_replace)
                    index_of_candidate_to_replace -= 1
            else:
                self.log(warning_text, level=2, prefix='  > ')

    def generate_candidate_individuals(self) -> None:
        '''
        Generate one trial vector per parent in the truncated valid archive.
        '''
        n_from_surrogate = max(self.problem.n_objective,
                            int(self.population_size * self.ratio_from_surrogate))
        
        self._generate_candidate_individuals_from_evolutionary_operators()
        
        self._generate_candidate_individuals_from_surrogate(n_from_surrogate)

    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)
