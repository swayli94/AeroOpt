'''
Surrogate-based Optimization (SBO).

Every candidate of an SBO iteration comes from an optimization run on the
surrogate model: the surrogate is retrained on the valid database, an inner
optimizer searches its adaptive-sampling criteria, and the resulting non-dominated
designs are handed to the expensive evaluator.
'''

from __future__ import annotations

from typing import Callable

import numpy as np

from aeroopt.core import Problem, Individual, MultiProcessEvaluation
from aeroopt.optimization.settings import SettingsOptimization
from aeroopt.optimization.base import OptBaseFramework, PostProcess
from aeroopt.optimization.hybrid.base import SurrogateOptimizationBase
from aeroopt.optimization.moea import DominanceBasedAlgorithm
from aeroopt.utils.surrogate import SurrogateModel


class PostProcessSBO(PostProcess):
    '''
    Report how well the surrogate predicted the candidates that were just
    evaluated for real.

    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object; must be an :class:`SBO` instance.
    surrogate: SurrogateModel
        The surrogate model being assessed.
    '''
    def __init__(self, opt: OptBaseFramework, surrogate: SurrogateModel):

        super().__init__(opt)

        self.surrogate = surrogate

    def apply(self) -> None:
        '''
        Compare surrogate predictions against the true values of `db_candidate`.
        '''
        if self.surrogate.size <= 0 or self.opt.db_candidate.size <= 0:
            self.opt.log(
                'Surrogate performance: skipped (model not trained yet or no candidates).',
                level=2, prefix='    ')
            return None

        self.opt.log('Evaluating the performance of the surrogate model.', level=1)

        if not isinstance(self.opt, SBO):
            raise TypeError('PostProcessSBO can only be used with SBO.')

        xs = self.opt.db_candidate.get_xs(scale=False)
        ys_actual = self.opt.db_candidate.get_ys(scale=False)
        ys_actual = ys_actual[:, self.opt.index_outputs_for_surrogate]

        performance = self.surrogate.evaluate_performance(xs, ys_actual)
        self.opt.log(f'RMSE: {performance["RMSE"]}', level=2, prefix='    ')


class SBO(SurrogateOptimizationBase):
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
        Optimization object run on the surrogate model.
    user_func: Callable
        User-defined function to evaluate the individuals, i.e. the expensive
        one. If None, use external evaluation script.
    user_func_supports_parallel: bool
        Whether `user_func` takes the whole `xs` matrix at once. This concerns
        the expensive evaluator only; the inner optimizer's own flag is set
        internally when it is pointed at the surrogate.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.

    Attributes:
    -----------
    outputs_for_surrogate: List[str]
        Names of the outputs predicted by the surrogate model.
    index_outputs_for_surrogate: np.ndarray
        Indices of those outputs within the global problem's output vector.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            surrogate: SurrogateModel,
            opt_on_surrogate: OptBaseFramework,
            user_func: Callable|None = None,
            user_func_supports_parallel: bool = False,
            mp_evaluation: MultiProcessEvaluation|None = None,
            save_result_files: bool = True,
            logging: bool = True,
            rng: np.random.Generator|None = None):

        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            surrogate=surrogate,
            opt_on_surrogate=opt_on_surrogate,
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
            rng=rng,
        )

    def generate_candidate_individuals(self) -> None:
        '''
        Fill `db_candidate` with the best designs found on the surrogate model.
        '''
        self._run_optimization_on_surrogate()

        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            self._population_database_of_surrogate(), self.population_size)

        self.db_candidate.empty_database()

        for parent in temp_parents.individuals:

            indi = Individual(problem=self.problem, x=parent.x)
            indi.source = 'surrogate_prediction'
            indi.generation = self.iteration

            added, warning_text = self.db_candidate.add_individual(
                indi,
                check_duplication=True,
                check_bounds=True,
                deepcopy=False,
                print_warning_info=False,
            )

            if not added:
                self.log(warning_text, level=2, prefix='  > ')
