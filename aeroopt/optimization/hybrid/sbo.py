'''
Surrogate-based Optimization (SBO)
'''
from typing import List, Callable, Tuple

import numpy as np
import functools

from aeroopt.core import (
    Problem, Individual, 
    MultiProcessEvaluation
)
from aeroopt.optimization.settings import SettingsOptimization
from aeroopt.optimization.base import OptBaseFramework, PreProcess, PostProcess
from aeroopt.utils.surrogate import SurrogateModel
from aeroopt.optimization.moea import DominanceBasedAlgorithm


def _surrogate_user_func(
    xs: np.ndarray,
    surrogate: SurrogateModel,
    **kwargs,
    ) -> Tuple[List[bool], np.ndarray]:
    ys = surrogate.predict_for_adaptive_sampling(np.asarray(xs, dtype=float), **kwargs)
    return [True] * len(xs), np.asarray(ys, dtype=float)


class PostProcessSBO(PostProcess):
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
        '''
        if self.surrogate.size <= 0 or self.opt.db_candidate.size <= 0:
            self.opt.log(
                'Surrogate performance: skipped (model not trained yet or no candidates).',
                level=2, prefix='    ')
            return None

        self.opt.log(f'Evaluating the performance of the surrogate model.', level=1)

        xs = self.opt.db_candidate.get_xs(scale=False)
        ys_actual = self.opt.db_candidate.get_ys(scale=False)
        ys_actual = ys_actual[:, self.opt.index_outputs_for_surrogate]

        performance_dict = self.surrogate.evaluate_performance(xs, ys_actual)
        for key, value in performance_dict.items():
            self.opt.log(f'{key:10s}: {value}', level=2, prefix='    ')


class SBO(OptBaseFramework):
    '''
    Surrogate-based Optimization (SBO).
    
    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    surrogate: SurrogateModel
        Surrogate model for optimization.
    opt_on_surrogate: OptBaseFramework
        Optimization object on the surrogate model.
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
        
    Attributes:
    -----------
    outputs_for_surrogate: List[str]
        List of names of outputs that are predicted by the surrogate model.
    index_outputs_for_surrogate: np.ndarray
        Indices of the output variables that are predicted by the surrogate model.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            surrogate: SurrogateModel,
            opt_on_surrogate: OptBaseFramework,
            user_func: Callable = None,
            mp_evaluation: MultiProcessEvaluation = None,
            pre_process: PreProcess = None,
            post_process: PostProcessSBO = None):
        
        super().__init__(problem, optimization_settings, user_func, mp_evaluation)
        
        self.surrogate = surrogate
        self.opt_on_surrogate = opt_on_surrogate
        
        self.pre_process = pre_process
        self.post_process = post_process
        
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
    
    def generate_candidate_individuals(self) -> None:
        '''
        Generate candidate individuals using the surrogate model.
        '''
        self.opt_on_surrogate.initialize()

        # Set the user function for the surrogate model
        user_func = functools.partial(_surrogate_user_func, surrogate=self.surrogate)
        self.opt_on_surrogate.user_func = user_func
        self.opt_on_surrogate.user_func_supports_parallel = True
        
        # Optimization on the surrogate model
        self.opt_on_surrogate.main()
        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            self.opt_on_surrogate.db_valid, self.population_size)
        n_pop = temp_parents.size
        
        # Get the candidate individuals from the surrogate model
        self.db_candidate.empty_database()
        
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
            else:
                self.log(warning_text, level=2, prefix='  > ')

    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)
