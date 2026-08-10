'''
Surrogate-assisted Optimization (SAO).

Unlike SBO, an SAO iteration mixes two candidate sources: an ordinary
evolutionary step on the real archive ("E" individuals) and an optimization run
on the surrogate model ("S" individuals). The surrogate therefore assists the
search instead of replacing it, which keeps progress going while the model is
still inaccurate.
'''

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from aeroopt.core import Problem, Individual, MultiProcessEvaluation
from aeroopt.optimization.settings import SettingsOptimization, SettingsDE
from aeroopt.optimization.base import OptBaseFramework, PostProcess
from aeroopt.optimization.hybrid.base import SurrogateOptimizationBase
from aeroopt.optimization.moea import DominanceBasedAlgorithm
from aeroopt.optimization.stochastic.de import DiffEvolution
from aeroopt.utils.surrogate import SurrogateModel

SOURCE_EVOLUTIONARY = 'evolutionary_operator'
SOURCE_SURROGATE = 'surrogate_prediction'


class PostProcessSAO(PostProcess):
    '''
    Report both the accuracy and the contribution of the surrogate model.

    Accuracy is the prediction error on the candidates that were just evaluated
    for real. Contribution is how many of the candidates on the current
    non-dominated front came from the surrogate ("S") rather than from the
    evolutionary operator ("E").

    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object; must be an :class:`SAO` instance.
    surrogate: SurrogateModel
        The surrogate model being assessed.
    '''
    def __init__(self, opt: OptBaseFramework, surrogate: SurrogateModel):

        super().__init__(opt)

        self.surrogate = surrogate

    def apply(self) -> None:
        '''
        Log the surrogate's prediction error and its share of the candidate front.
        '''
        if self.surrogate.size <= 0 or self.opt.db_candidate.size <= 0:
            self.opt.log(
                'Surrogate performance: skipped (model not trained yet or no candidates).',
                level=2, prefix='    ')
            return None

        self.opt.log('Evaluating the performance and contribution of the surrogate model.',
                     level=1)

        if not isinstance(self.opt, SAO):
            raise TypeError('PostProcessSAO can only be used with SAO.')

        xs = self.opt.db_candidate.get_xs(scale=False)
        ys_actual = self.opt.db_candidate.get_ys(scale=False)
        ys_actual = ys_actual[:, self.opt.index_outputs_for_surrogate]

        index_by_source = {SOURCE_EVOLUTIONARY: [], SOURCE_SURROGATE: []}
        for i, indi in enumerate(self.opt.db_candidate.individuals):
            if indi.source in index_by_source:
                index_by_source[indi.source].append(i)

        index_e = np.array(index_by_source[SOURCE_EVOLUTIONARY], dtype=int)
        index_s = np.array(index_by_source[SOURCE_SURROGATE], dtype=int)

        self.opt.log(f'Number of "S" individuals (from surrogate predictions):  {index_s.size}',
                     level=2, prefix='    ')
        self.opt.log(f'Number of "E" individuals (from evolutionary operators): {index_e.size}',
                     level=2, prefix='    ')

        #* Prediction accuracy per source.
        # `evaluate_performance` averages over rows, so an empty group has
        # nothing to report and would produce NaNs.
        for label, index in (('S', index_s), ('E', index_e)):
            if index.size == 0:
                continue
            performance = self.surrogate.evaluate_performance(
                xs[index], ys_actual[index])
            self.opt.log(f'Surrogate performance on "{label}" individuals:',
                         level=2, prefix='    ')
            self.opt.log(f'RMSE: {performance["RMSE"]}', level=2, prefix='    ')

        #* Contribution: sources of the candidates' first non-dominated front.
        temp_db = copy.deepcopy(self.opt.db_candidate)
        temp_db.eliminate_invalid_individuals()
        index_fronts = DominanceBasedAlgorithm.non_dominated_ranking(temp_db)

        if not index_fronts or not index_fronts[0]:
            self.opt.log('No feasible candidate to form a Pareto front.',
                         level=2, prefix='    ')
            return None

        n_by_source = {SOURCE_EVOLUTIONARY: 0, SOURCE_SURROGATE: 0}
        for i in index_fronts[0]:
            source = temp_db.individuals[i].source
            if source in n_by_source:
                n_by_source[source] += 1

        self.opt.log('Sources of Pareto front of the candidates:', level=2, prefix='    ')
        self.opt.log(f'Number of "S" individuals: {n_by_source[SOURCE_SURROGATE]}',
                     level=2, prefix='    ')
        self.opt.log(f'Number of "E" individuals: {n_by_source[SOURCE_EVOLUTIONARY]}',
                     level=2, prefix='    ')


class SAO(SurrogateOptimizationBase):
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
        Optimization object run on the surrogate model.
    ratio_from_surrogate: float
        Fraction of the population that is replaced by surrogate-derived candidates.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    rng: np.random.Generator
        Random generator for differential evolution in the main loop.
        If None, one is derived from `optimization_settings.seed`.

    Attributes:
    -----------
    outputs_for_surrogate: List[str]
        Names of the outputs predicted by the surrogate model.
    index_outputs_for_surrogate: np.ndarray
        Indices of those outputs within the global problem's output vector.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsDE,
            surrogate: SurrogateModel,
            opt_on_surrogate: OptBaseFramework,
            ratio_from_surrogate: float = 0.5,
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

        self.algorithm_settings = algorithm_settings
        self.ratio_from_surrogate = ratio_from_surrogate

    def _generate_candidate_individuals_from_evolutionary_operators(self) -> None:
        '''
        Fill `db_candidate` with DE offspring of the real archive.
        '''
        DiffEvolution.generate_candidate_individuals(
            db=self.select_population_database(),
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            scale_factor=self.algorithm_settings.scale_factor,
            cross_rate=self.algorithm_settings.cross_rate,
            rng=self.rng,
        )

    def _generate_candidate_individuals_from_surrogate(self, n_candidates: int) -> None:
        '''
        Append up to `n_candidates` surrogate-derived designs to `db_candidate`.

        `db_candidate` is not emptied: each accepted surrogate candidate evicts
        one evolutionary candidate from the tail, so the population size stays
        at `population_size` while the mix shifts towards the surrogate.
        '''
        index_of_candidate_to_replace = self.db_candidate.size - 1

        self._run_optimization_on_surrogate()

        temp_parents = DominanceBasedAlgorithm.build_temporary_parent_database(
            self._population_database_of_surrogate(), n_candidates)

        for parent in temp_parents.individuals:

            indi = Individual(problem=self.problem, x=parent.x)
            indi.source = SOURCE_SURROGATE
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
                continue

            # Keep the size of db_candidate equal to population_size by
            # dropping one of the original evolutionary candidates. Stop
            # evicting once they are exhausted, so surrogate candidates are
            # never removed and the index never goes negative.
            if self.db_candidate.size > self.population_size:
                if index_of_candidate_to_replace < 0:
                    break
                self.db_candidate.delete_individual(index=index_of_candidate_to_replace)
                index_of_candidate_to_replace -= 1

    def generate_candidate_individuals(self) -> None:
        '''
        Build the candidate population from both sources: DE offspring first,
        then surrogate-derived designs replacing part of them.
        '''
        n_from_surrogate = max(self.problem.n_objective,
                            int(self.population_size * self.ratio_from_surrogate))

        self._generate_candidate_individuals_from_evolutionary_operators()

        self._generate_candidate_individuals_from_surrogate(n_from_surrogate)
