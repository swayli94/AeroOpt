'''
Shared plumbing for surrogate-driven optimization frameworks (SAO and SBO).
'''

from __future__ import annotations

import functools
from typing import Callable, List, Tuple

import numpy as np

from aeroopt.core import Problem, Database, MultiProcessEvaluation
from aeroopt.optimization.base import (
    OptBaseFramework, select_population_database,
)
from aeroopt.optimization.settings import SettingsOptimization
from aeroopt.utils.surrogate import SurrogateModel


def surrogate_user_func(
        xs: np.ndarray,
        surrogate: SurrogateModel,
        **kwargs,
        ) -> Tuple[List[bool], np.ndarray]:
    '''
    Adapt a surrogate model to the batch `user_func` protocol.

    The inner optimizer treats the surrogate's adaptive-sampling criteria as if
    they were real objectives, so every prediction is reported as successful.

    Parameters:
    -----------
    xs: np.ndarray [n, n_input]
        Input matrix to predict.
    surrogate: SurrogateModel
        Trained surrogate model.

    Returns:
    --------
    list_succeed: List[bool]
        All True; a prediction never fails.
    ys: np.ndarray [n, n_output_of_surrogate]
        Adaptive-sampling criteria.
    '''
    ys = surrogate.predict_for_adaptive_sampling(np.asarray(xs, dtype=float), **kwargs)
    return [True] * len(xs), np.asarray(ys, dtype=float)


class SurrogateOptimizationBase(OptBaseFramework):
    '''
    Base class for optimization frameworks that drive an inner optimizer over a
    surrogate model.

    It owns the mapping between the surrogate's outputs and the outputs of the
    global problem, retrains the surrogate at the start of each iteration, and
    runs the inner optimizer on the surrogate's predictions.

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

    Other parameters are forwarded to :class:`OptBaseFramework`.

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
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
            rng=rng,
        )

        self.surrogate = surrogate
        self.opt_on_surrogate = opt_on_surrogate

        self._set_outputs_for_surrogate()

    @property
    def outputs_for_surrogate(self) -> List[str]:
        '''
        Names of the outputs that are predicted by the surrogate model.
        '''
        return self._outputs_for_surrogate

    @property
    def index_outputs_for_surrogate(self) -> np.ndarray:
        '''
        Indices of the surrogate's outputs within the global problem's outputs.

        Used to slice the global output matrix down to the surrogate's columns.
        '''
        return self._index_outputs_for_surrogate

    def _set_outputs_for_surrogate(self) -> None:
        '''
        Resolve the surrogate's output names against the global problem.
        '''
        self._outputs_for_surrogate = self.surrogate.problem.name_output

        try:
            self._index_outputs_for_surrogate = np.array(
                [self.problem.name_output.index(name)
                 for name in self._outputs_for_surrogate], dtype=int)

        except ValueError as e:
            self.log('Outputs defined in the surrogate model are not in the global optimization problem.',
                        level=2, prefix='    ')
            self.log(f'Error message: {e}', level=2, prefix='    ')
            raise ValueError(
                'Surrogate outputs must be a subset of the problem outputs: '
                f'{self._outputs_for_surrogate} vs {self.problem.name_output}') from e

    #TODO: Can be adapted
    def update_parameters(self) -> None:
        '''
        Retrain the surrogate model on the valid database.

        Called once per iteration, before candidates are generated.
        '''
        if self.db_valid.size <= 0:
            self.log('Surrogate training skipped: the valid database is empty.',
                     level=2, prefix='    ')
            return

        xs = self.db_valid.get_xs(scale=False)
        ys = self.db_valid.get_ys(scale=False)
        ys = ys[:, self.index_outputs_for_surrogate]
        self.surrogate.train(xs, ys)

    def _run_optimization_on_surrogate(self) -> None:
        '''
        Run the inner optimizer with the surrogate as its evaluation function.

        The inner optimizer is reset first, so each outer iteration searches the
        freshly retrained surrogate from scratch.
        '''
        self.opt_on_surrogate.initialize()

        self.opt_on_surrogate.user_func = functools.partial(
            surrogate_user_func, surrogate=self.surrogate)
        self.opt_on_surrogate.user_func_supports_parallel = True

        self.opt_on_surrogate.main()

    def _population_database_of_surrogate(self) -> Database:
        '''
        Population database of the inner optimizer, using the same
        valid-vs-total rule as :meth:`OptBaseFramework.select_population_database`.

        Only the inner optimizer's `db_valid` and `db_total` are required, so a
        lightweight stand-in can be plugged in as `opt_on_surrogate`.
        '''
        return select_population_database(
            self.opt_on_surrogate.db_valid,
            self.opt_on_surrogate.db_total,
            self.population_size)
