'''
Base framework for optimization.
'''

from __future__ import annotations

import os
import numpy as np
import time
from abc import ABC, abstractmethod

from typing import List, Callable, Tuple

from aeroopt.core import (
    Problem, Individual, Database,
    MultiProcessEvaluation,
    init_log, log
)
from aeroopt.optimization.settings import (
    SettingsOptimization, SettingsGeneticOperators,
)
from aeroopt.optimization.moea import DominanceBasedAlgorithm
from aeroopt.analysis.analyze_database import AnalyzeDatabase


#* An archive smaller than this is considered too small to drive evolution on
#* its own, so the (larger, possibly infeasible) total database is used instead.
MIN_VALID_ARCHIVE_SIZE = 5
MIN_VALID_ARCHIVE_RATIO = 0.5


def select_population_database(db_valid: Database, db_total: Database,
                               population_size: int) -> Database:
    '''
    Choose the database that offspring are generated from.

    The valid archive holds only feasible, successfully evaluated individuals
    and is the natural parent pool. Early in a run (or on a heavily constrained
    problem) it can be too small to support meaningful selection and variation,
    so the total database is used instead --- its individuals are ranked with
    the constraint-aware dominance rules of
    :meth:`~aeroopt.core.individual.Individual.check_dominance`, which prefer
    feasible over infeasible and lower total violation among infeasible ones.

    Parameters:
    -----------
    db_valid: Database
        Archive of feasible individuals.
    db_total: Database
        Archive of all individuals, including infeasible ones.
    population_size: int
        Population size of the optimization.

    Returns:
    --------
    db: Database
        `db_valid` when it is large enough, otherwise `db_total`.
    '''
    threshold = max(MIN_VALID_ARCHIVE_SIZE,
                    int(population_size * MIN_VALID_ARCHIVE_RATIO))

    if db_valid.size <= threshold:
        return db_total

    return db_valid


class OptBaseFramework(ABC):
    '''
    Base framework for optimization.

    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    user_func_supports_parallel: bool
        If True, the user-defined function inherently supports parallel evaluation,
        i.e., `list_succeed, ys = user_func(xs, **kwargs)` can be directly called in this function.
        If False, either use `mp_evaluation` for parallel evaluation, or use serial evaluation.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    rng: np.random.Generator|None
        Random generator used by the evolutionary operators. If None, one is
        created from `optimization_settings.seed`, so a seed in the settings
        file makes the whole run reproducible.

    Attributes:
    -----------
    iteration: int
        The current iteration number.
    rng: np.random.Generator
        Random generator used by the evolutionary operators.
    pre_process: PreProcess|None
        Pre-processing of the `db_candidate` database to be evaluated.
    post_process: PostProcess|None
        Post-processing of the `db_candidate` database after evaluation.
    db_total: Database
        Total database, containing all individuals.
    db_valid: Database
        Valid database, containing all the feasible individuals.
    db_elite: Database
        Elite database, containing the elite individuals, e.g., Pareto-optimal solutions.
    db_candidate: Database
        Population database, containing the candidate individuals,
        e.g., initial population, offspring individuals, etc.
    analyze_total: AnalyzeDatabase
        Analysis of the total database to:
        (1) avoid having duplicated individuals in `db_candidate`;
        (2) find new candidates using potential-based search.
    analyze_valid: AnalyzeDatabase
        Analysis of the valid database to:
        (1) adjust candidate input variables to be feasible.

    Example:
    ---------
    >>> def user_func(x: np.ndarray, **kwargs) -> Tuple[bool, np.ndarray]:
    >>>     return True, np.array([np.sum(x**2)])
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            user_func: Callable|None = None,
            user_func_supports_parallel: bool = False,
            mp_evaluation: MultiProcessEvaluation|None = None,
            save_result_files: bool = True,
            logging: bool = True,
            rng: np.random.Generator|None = None):

        self.problem = problem
        self.optimization_settings = optimization_settings

        self.user_func : Callable|None = user_func
        self.mp_evaluation : MultiProcessEvaluation|None = mp_evaluation
        self.user_func_supports_parallel : bool = user_func_supports_parallel
        self.iteration : int = 0
        self.save_result_files : bool = save_result_files
        self.logging : bool = logging

        # Next ID handed to a candidate. It only ever moves forward, because an
        # ID names the external working folder: reusing one would make the run
        # read the previous design's results. `db_total.get_largest_ID() + 1` is
        # not enough on its own --- a candidate rejected as a duplicate never
        # reaches `db_total`, so its ID would be handed out a second time.
        self._next_ID : int = 1

        self.rng : np.random.Generator = (
            rng if rng is not None
            else np.random.default_rng(optimization_settings.seed))


        # Processing objects manually defined in the main program.
        self.pre_process : PreProcess|None = None
        self.post_process : PostProcess|None = None

        # Database
        self.db_total = Database(self.problem, database_type='total')
        self.db_valid = Database(self.problem, database_type='valid')
        self.db_elite = Database(self.problem, database_type='elite')
        self.db_candidate = Database(self.problem, database_type='population')

        # Analysis of the database
        self.analyze_total = AnalyzeDatabase(self.db_total,
                               variables_for_calculating_potential=None,
                               critical_potential=self.optimization_settings.critical_potential_x)

        self.analyze_valid = AnalyzeDatabase(self.db_valid,
                               variables_for_calculating_potential=None,
                               critical_potential=self.optimization_settings.critical_potential_x)

        # Attributes
        self._start_time = time.perf_counter()

        if self.logging:
            init_log(self.dir_summary, self.fname_log)
            self.log(f'Optimization [{self.name}] initialized.', level=0, prefix='=== ')

    @property
    def population_size(self) -> int:
        '''
        Number of individuals in the population.
        '''
        return self.optimization_settings.population_size

    @property
    def max_iterations(self) -> int:
        '''
        Maximum number of iterations in the optimization.
        '''
        return self.optimization_settings.max_iterations

    @property
    def name(self) -> str:
        '''
        Name of the optimization, i.e.,
        `{OptimizationName}-{ProblemName}`
        '''
        return self.optimization_settings.name + '-' + self.problem.name

    @property
    def dir_save(self) -> str:
        '''
        Directory to save the results, i.e., `Calculation` folder.
        '''
        return os.path.join(self.optimization_settings.working_directory, 'Calculation')

    @property
    def dir_summary(self) -> str:
        '''
        Directory to save the summary of the optimization, i.e., `Summary` folder.
        '''
        return os.path.join(self.optimization_settings.working_directory, 'Summary')

    @property
    def dir_runfiles(self) -> str:
        '''
        Directory of the `Runfiles` folder, which contains the external evaluation script.
        '''
        return os.path.join(self.optimization_settings.working_directory, 'Runfiles')

    @property
    def fname_log(self) -> str:
        '''
        Name of the log file defined in the optimization settings.
        '''
        return os.path.join(self.optimization_settings.working_directory,
                            self.optimization_settings.fname_log)

    @property
    def fname_db_total(self) -> str:
        '''
        Name of the total database file.
        '''
        return os.path.join(self.dir_summary, self.optimization_settings.fname_db_total)

    @property
    def fname_db_elite(self) -> str:
        '''
        Name of the elite database file.
        '''
        return os.path.join(self.dir_summary, self.optimization_settings.fname_db_elite)

    @property
    def level(self) -> int:
        '''
        Level of the information to be printed on the screen.

        The text will be printed on the screen if its level <= self.level.
        '''
        return self.optimization_settings.info_level_on_screen

    @property
    def max_ID(self) -> int:
        '''
        Maximum ID of the individuals in the total database.
        '''
        return self.db_total.get_largest_ID()

    def select_population_database(self) -> Database:
        '''
        Choose the database that offspring are generated from.

        See the module-level :func:`select_population_database` for the rule.

        Returns:
        --------
        db: Database
            `db_valid` when it is large enough, otherwise `db_total`.
        '''
        return select_population_database(
            self.db_valid, self.db_total, self.population_size)

    def initialize(self) -> None:
        '''
        Initialize the optimization to start a new optimization.
        '''
        self.db_total.empty_database()
        self.db_valid.empty_database()
        self.db_elite.empty_database()
        self.db_candidate.empty_database()
        self.iteration = 0
        self._next_ID = 1
        self._start_time = time.perf_counter()

        self.log(f'Optimization [{self.name}] initialized.', level=0, prefix='=== ')

    #* Main procedures

    def main(self) -> None:
        '''
        Main loop of the optimization.
        '''
        self._warn_about_existing_case_folders()

        self.resume()

        self.initialize_population()

        self.select_elite_from_valid()

        self.save_results()

        while not self.termination():

            self.iteration += 1
            t0 = time.perf_counter()
            self.log(f'Iteration {self.iteration} started.', level=1, prefix='=== ')

            self.update_parameters()

            self.generate_candidate_individuals()

            if self.pre_process is not None:
                self.pre_process.apply()

            self.evaluate_db_candidate()

            self.update_total_and_valid_with_candidate()

            if self.post_process is not None:
                self.post_process.apply()
                self.derive_valid_from_total()

            self.select_elite_from_valid()

            self.save_results()

            t1 = time.perf_counter()
            self.log(f'Iteration {self.iteration} finished in {(t1-t0)/60.0:.2f} min.', level=1)

        time_elapsed = time.perf_counter() - self._start_time
        self.log(f'Optimization [{self.name}] finished in {time_elapsed/60.0:.2f} min.', level=0, prefix='=== ')

    def resume(self) -> None:
        '''
        Resume the optimization from previous results.
        '''
        if not self.optimization_settings.resume:
            return None

        fname = os.path.join(self.dir_summary, self.optimization_settings.fname_db_resume)

        self.db_total.read_database_json(fname)
        self.db_total.update_id_list()

        for indi in self.db_total.individuals:
            indi.generation = 0
            indi.source = 'previous_database'

        self.iteration = 0

        # Continue numbering past the resumed database, so the new candidates do
        # not evaluate in the working folders of the previous run.
        self._next_ID = self.max_ID + 1

        self.log(f'Resume from [{fname}], size = {self.db_total.size}.', level=0)

    def initialize_population(self) -> None:
        '''
        Initialize the initial population `db_candidate` database:

        - generate initial individuals (Design of Experiments, perturbation, user-defined, etc.)
        - pre-processing of `db_candidate`
        - evaluation of `db_candidate`
        - update `db_total` and `db_valid`
        - post-processing of `db_total` and `db_valid`
        '''
        self.log('Initial population preparation started.', level=1)

        self.generate_initial_individuals()

        if self.db_candidate.size > 0:

            if self.pre_process is not None:
                self.pre_process.apply()

            self.evaluate_db_candidate()

        self.update_total_and_valid_with_candidate()

        if self.post_process is not None:
            self.post_process.apply()
            self.derive_valid_from_total()

        self.log(f"Initial population prepared: valid={self.db_valid.size}.", level=1)

    #TODO: Can be adapted
    def generate_initial_individuals(self) -> None:
        '''
        Generate the initial individuals for optimization.

        - this is the default implementation with random sampling.
        - can be adapted to other methods, e.g., Design of Experiments, perturbation, user-defined, etc.
        - the initial individuals are stored in `db_candidate` database.

        A study that resumed a database already has an initial population, so
        none is sampled: the point of resuming is to carry on from those
        designs, not to spend another `population_size` evaluations on a fresh
        sample of the same space (with a fixed `seed`, on the *identical*
        sample). Set `force_initial_population_size` to sample anyway, e.g. to
        widen a converged archive.
        '''
        # xs = np.random.rand(self.population_size, self.problem.n_input)
        # xs = self.problem.scale_x(xs, reverse=True)

        if self.optimization_settings.force_initial_population_size is not None:
            population_size = self.optimization_settings.force_initial_population_size

        elif self.db_total.size > 0:
            self.db_candidate.empty_database()
            self.log(f'Resumed {self.db_total.size} individuals; no initial sample taken. '
                     'Set `force_initial_population_size` to sample anyway.', level=1)
            return

        else:
            population_size = self.population_size

        if population_size <= 0:
            self.db_candidate.empty_database()
            self.log('Initial population size is set to 0.', level=1)
            return

        xs = self.problem.latin_hypercube_sampling(population_size,
                                    scaled_values=False,
                                    sample_variables=None,
                                    seed=self.optimization_settings.seed)

        self.db_candidate.empty_database()
        for x in xs:
            indi = Individual(problem=self.problem, x=x)
            indi.source = 'DoE'
            indi.generation = 0
            added, warning_info = self.db_candidate.add_individual(indi, check_duplication=True,
                                    check_bounds=True, deepcopy=False, print_warning_info=False)
            if not added:
                self.log(warning_info, level=2, prefix='  - ')

    #TODO: Can be adapted
    def termination(self) -> bool:
        '''
        Check if the optimization should be terminated.
        '''
        return self.iteration >= self.max_iterations

    #TODO: Can be adapted
    def update_parameters(self) -> None:
        '''
        Update settings and parameters of the optimization.
        '''
        return None

    #TODO: Can be adapted
    def save_results(self) -> None:
        '''
        Save the results of the optimization.
        '''
        if not self.save_result_files:
            return

        os.makedirs(self.dir_summary, exist_ok=True)
        self.db_total.output_database_json(self.fname_db_total)
        self.db_elite.output_database_json(self.fname_db_elite)

    @abstractmethod
    def generate_candidate_individuals(self) -> None:
        '''
        Generate candidate individuals during the optimization,
        which are stored in `db_candidate` database before evaluation.
        The `db_candidate` database is generated from `db_valid` database:

        - create a temporary parent database by selection from `db_valid`
        - evolution (crossover, mutation, etc.) of the parent database
        - add user-defined new individuals
        - search new candidates from surrogate models
        '''
        pass

    def evaluate_db_candidate(self) -> None:
        '''
        Evaluate the `db_candidate` database,
        then add the individuals to `db_total`.

        The candidates are snapped to the precision grid, screened against
        everything already evaluated, and given run-unique IDs first, so that
        what is evaluated is a legal design that has not been paid for before
        and its results land in a working folder of its own. All three steps
        happen here, after the pre-processing hook, so that candidates a hook
        added or moved are covered too.
        '''
        self._apply_precision_to_candidate_individuals()

        self._drop_candidates_already_evaluated()

        self._assign_ID_to_candidate_individuals()

        if self.db_candidate.size <= 0:
            self.log('No candidate left to evaluate.', level=1)
            return

        t0 = time.perf_counter()

        self.db_candidate.evaluate_individuals(mp_evaluation=self.mp_evaluation,
                                user_func=self.user_func,
                                user_func_supports_parallel=self.user_func_supports_parallel)

        t1 = time.perf_counter()
        self.log(f'Evaluation of {self.db_candidate.size} candidates finished in {(t1-t0)/60.0:.2f} min.', level=1)

    def update_total_and_valid_with_candidate(self) -> None:
        '''
        Update the total database and valid database with the candidate database.
        - Merge the candidate database into the total database.
        - Copy the total database to the valid database.
        - Eliminate invalid individuals from the valid database.
        '''
        n_previous_total = self.db_total.size
        n_previous_valid = self.db_valid.size

        for indi in self.db_candidate.individuals:
            indi.generation = self.iteration

        self.db_total.merge_with_database(
            self.db_candidate, deepcopy=True, log_func=self.log)

        self.derive_valid_from_total()

        n_added_total = self.db_total.size - n_previous_total
        n_added_valid = self.db_valid.size - n_previous_valid

        self.log(f'Add {n_added_total} individuals to total, updated to {self.db_total.size}.',
                    level=1, prefix='    ')
        self.log(f'Add {n_added_valid} individuals to valid, updated to {self.db_valid.size}.',
                    level=1, prefix='    ')

    def derive_valid_from_total(self) -> None:
        '''
        Rebuild `db_valid` as the feasible, successfully evaluated part of
        `db_total`.

        `db_valid` is derived, never maintained in place, because a constraint
        can depend on an output and a post-processing hook can change what
        counts as feasible --- or prune `db_total` outright. It is therefore
        re-derived *after* the hook as well as after the merge; otherwise an
        individual the hook removed would still be picked as elite and written
        to the summary for that iteration.
        '''
        self.db_valid.copy_from_database(self.db_total, deepcopy=True)
        self.db_valid.eliminate_invalid_individuals()

        # `db_total` / `db_valid` objects are not reassigned here (in-place merge + copy_into),
        # but keep analyzers pinned to the canonical databases in case they were ever pointed
        # elsewhere (e.g. stale `db_candidate`, or legacy code that swapped `db_valid`).
        if self.analyze_total is not None:
            self.analyze_total.database = self.db_total
        if self.analyze_valid is not None:
            self.analyze_valid.database = self.db_valid

    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database into `db_elite`.

        The default is Pareto-dominance ranking plus crowding-distance
        assignment, which stores the first non-dominated front. Override this
        to use a different elite criterion (e.g. an indicator-based one).
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)

    #* Support functions

    def log(self, text: str, level: int = 1, prefix: str = '>>> ') -> None:
        '''
        Log a message to the log file.
        '''
        if not self.logging:
            return

        log(text, prefix=prefix, fname=self.fname_log,
                print_on_screen=(level<=self.level))

    def _assign_ID_to_candidate_individuals(self) -> None:
        '''
        Assign IDs to the individuals in `db_candidate` that are unique for the
        whole run.

        A candidate's ID is the name of its external working folder
        (`Calculation/<ID>`), and it is the ID the individual keeps once it is
        merged into `db_total`. `db_candidate` is emptied and refilled every
        iteration, so its own IDs restart at 1 each time; without this step the
        second iteration would evaluate in the folders of the first.
        '''
        id_next = max(self._next_ID, self.max_ID + 1)

        for i in range(self.db_candidate.size):
            self.db_candidate.individuals[i].ID = id_next + i

        self._next_ID = id_next + self.db_candidate.size

        self.db_candidate.update_id_list()

    def _warn_about_existing_case_folders(self) -> None:
        '''
        Warn when the external calculation folder already holds cases.

        A fresh study numbers its cases from 1 again, so those folders will be
        hit by the new candidates. `Problem.external_run` raises on the first
        one whose input file does not match, but that happens once the study is
        already running --- this says it up front, while clearing the folder is
        still cheap.
        '''
        if self.user_func is not None:
            return

        folder = self.problem.calculation_folder

        if not os.path.isdir(folder):
            return

        n_existing = len([entry for entry in os.listdir(folder)
                          if os.path.isdir(os.path.join(folder, entry))])

        if n_existing == 0:
            return

        if self.optimization_settings.resume:
            self.log(f'Calculation folder [{folder}] holds {n_existing} case folders; '
                     'cases that match the resumed database are reused.', level=0)
        else:
            self.log(f'Calculation folder [{folder}] already holds {n_existing} case '
                     'folders from an earlier study, and a new study numbers its cases '
                     'from 1 again. Clear or move it, otherwise the run stops at the '
                     'first stale case.', level=0, prefix='!!! ')

    def _drop_candidates_already_evaluated(self) -> None:
        '''
        Remove candidates that duplicate a design already in `db_total`.

        `db_candidate` only ever checked for duplicates *within itself*: the
        check against everything evaluated so far happened at the merge, after
        the solver had already run. The duplicate was then discarded, so the
        evaluation bought nothing --- on a coarse precision grid, or once the
        population converges, that is a double-digit share of the budget.

        The same scaled distance decides here as at the merge
        (`critical_scaled_distance`), so exactly the candidates that would be
        rejected later are the ones dropped now.
        '''
        if self.db_candidate.size <= 0 or self.db_total.size <= 0:
            return

        # `get_xs` is always a matrix, so the list form comes back.
        is_duplicated, closest_index = self.db_total.check_duplication(
            self.db_candidate.get_xs(scale=True), is_scaled_x=True)

        keep = []
        for i, indi in enumerate(self.db_candidate.individuals):
            if is_duplicated[i]:
                closest_ID = self.db_total.individuals[closest_index[i]].ID
                self.log(f'Candidate #{i+1} duplicates evaluated ID '
                         f'{closest_ID}; not evaluated again.',
                         level=2, prefix='  - ')
            else:
                keep.append(indi)

        n_dropped = self.db_candidate.size - len(keep)

        if n_dropped > 0:
            self.db_candidate.individuals = keep
            self.db_candidate.update_id_list()
            self.log(f'Skipped {n_dropped} candidates already evaluated.',
                     level=1, prefix='    ')

    def _apply_precision_to_candidate_individuals(self) -> None:
        '''
        Snap every candidate's input vector to the precision grid.

        The operators already do this, so this is the safety net for candidates
        that reach `db_candidate` another way: a pre-processing hook, a
        user-defined injection, or a new algorithm whose author forgot.
        Evaluating an off-grid design wastes a solver run on a bridge script
        that rejects, say, a non-integer number of ribs.
        '''
        n_snapped = 0

        for indi in self.db_candidate.individuals:

            x = indi.x.copy()
            self.problem.apply_precision_x(x)

            if not np.array_equal(x, indi.x):
                indi.update_x(x)
                n_snapped += 1

        if n_snapped > 0:
            self.log(f'Snapped {n_snapped} candidates to the input precision grid.',
                     level=2, prefix='  - ')


class OptGeneticFramework(OptBaseFramework):
    '''
    Base class for drivers whose offspring come from the SBX / polynomial-mutation
    operator pair (NSGA-II, NSGA-III, RVEA, MOEA/D).

    It only adds the handling of the shared genetic-operator settings; the
    optimization loop is unchanged from :class:`OptBaseFramework`.

    Parameters:
    -----------
    algorithm_settings: SettingsGeneticOperators
        Crossover and mutation settings of the algorithm.

    Other parameters are forwarded to :class:`OptBaseFramework`.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsGeneticOperators,
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

        self.algorithm_settings = algorithm_settings

    @property
    def mut_rate_per_variable(self) -> float:
        '''
        Polynomial-mutation probability applied to each input variable.

        The configured `mut_rate` is the expected number of mutated variables
        per individual, so it is divided by the number of input variables.
        '''
        return self.algorithm_settings.mut_rate / max(self.problem.n_input, 1)


class PreProcess(ABC):
    '''
    Pre-processing of `db_candidate` database in each iteration.

    The databases are accessed through the `OptBaseFramework` object, `opt`.
    The `db_candidate` is modified in place.

    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object.
    '''
    def __init__(self, opt: OptBaseFramework):

        self.opt = opt

        self.pre_process_folder : str = 'PreProcess'

    @abstractmethod
    def apply(self) -> None:
        '''
        Apply the pre-processing to the `db_candidate` database.
        '''
        self.opt.log(f'Pre-processing of {self.opt.db_candidate.size} candidates started.', level=1)
        pass

    def _restrict_x_values_by_valid_database(self, xs: np.ndarray,
                        min_scaled_distance: float = 0.0,
                        max_scaled_distance: float = 1.0,
                        ID_list: List[int]|None = None,
                        ) -> np.ndarray:
        '''
        Restrict the input variables of candidates,
        so that their scaled distances to the valid individuals in `db_valid`
        are within [min_scaled_distance, max_scaled_distance].

        Parameters:
        -----------
        xs: np.ndarray [n_candidate, n_input]
            Input variables of the candidates.
        min_scaled_distance: float
            Minimum scaled distance to the valid individuals in `db_valid`.
        max_scaled_distance: float
            Maximum scaled distance to the valid individuals in `db_valid`.
        ID_list: List[int]|None
            List of local IDs of the `xs` to be restricted.
            If None, use index of `xs` as the list of IDs.

        Returns:
        --------
        xs_new: np.ndarray [n_candidate, n_input]
            Input variables of the candidates after restriction.
        '''
        n_candidate = xs.shape[0]

        if n_candidate <= 0 or self.opt.db_valid.size <= 0:
            return xs

        xs_new = np.zeros_like(xs)

        if ID_list is None:
            ID_list = list(range(n_candidate))

        scaled_xs = self.opt.problem.scale_x(xs)

        distance_matrix = self.opt.analyze_valid.calculate_distance_to_database(
                                scaled_xs, update_attributes=True) # [n_candidate, n_valid]

        min_distance = np.min(distance_matrix, axis=1) # [n_candidate]

        critical_scaled_distance = self.opt.problem.critical_scaled_distance
        min_scaled_distance = max(min_scaled_distance, critical_scaled_distance)
        max_scaled_distance = max(max_scaled_distance, critical_scaled_distance)

        for i in range(n_candidate):

            min_d = max(min_distance[i], 0.0)

            if min_d <= critical_scaled_distance:
                # Duplicated with a valid individual (lower than the problem's critical scaled distance).
                # Randomly select one from some nearest valid individuals (exclude the duplicated one)
                if self.opt.db_valid.size <= 1:
                    xs_new[i] = xs[i]
                    self.opt.log(f'Candidate #{ID_list[i]:2d}: duplicated with the only valid individual.',
                            level=2, prefix='  - ')
                    continue

                n_near = min(5, self.opt.db_valid.size)
                indices = np.argsort(distance_matrix[i])[1:n_near]
                j_valid = np.random.choice(indices)
                indi_ref = self.opt.analyze_valid.database.individuals[j_valid]

                _distance = distance_matrix[i, j_valid]
                _new_d = min_scaled_distance + np.random.uniform(0.2, 0.8)*(max_scaled_distance-min_scaled_distance)
                ratio = _new_d / _distance

                xs_new[i] = indi_ref.x + ratio * (xs[i] - indi_ref.x)

                self.opt.log(f'Candidate #{ID_list[i]:2d}: duplicated with the' +
                            f' nearest valid individual X (ID={indi_ref.ID:4d}),' +
                            f' adjust towards another valid individual by ratio {ratio:.2f}.',
                            level=2, prefix='  - ')
                continue

            elif min_d < min_scaled_distance:
                # Too close to the nearest valid individual.

                j_valid = np.argmin(distance_matrix[i])
                indi_ref = self.opt.analyze_valid.database.individuals[j_valid]
                ratio = min_scaled_distance / min_d
                xs_new[i] = indi_ref.x + ratio * (xs[i] - indi_ref.x)

                self.opt.log(f'Candidate #{ID_list[i]:2d}: too close to the' +
                            f' nearest valid individual X (ID={indi_ref.ID:4d}),' +
                            f' adjust distance by ratio {ratio:.2f} away from X.',
                            level=2, prefix='  - ')

            elif min_d > max_scaled_distance:
                # Too far from the nearest valid individual.

                j_valid = np.argmin(distance_matrix[i])
                indi_ref = self.opt.analyze_valid.database.individuals[j_valid]
                ratio = max_scaled_distance / min_d
                xs_new[i] = indi_ref.x + ratio * (xs[i] - indi_ref.x)

                self.opt.log(f'Candidate #{ID_list[i]:2d}: too far from the' +
                            f' nearest valid individual X (ID={indi_ref.ID:4d}),' +
                            f' adjust distance by ratio {ratio:.2f} towards X.',
                            level=2, prefix='  - ')

            else:

                xs_new[i] = xs[i]

        self.opt.problem.apply_bounds_x(xs_new)
        self.opt.problem.apply_precision_x(xs_new)

        return xs_new

    def _check_pre_processing_feasibility(self, xs: np.ndarray,
                        pre_processing_problem: Problem,
                        user_pre_processing_func: Callable|None = None) -> Tuple[List[bool], List[int]]:
        '''
        Check the feasibility of the input variables after pre-processing:
        - check individual's `valid_evaluation` flag
        - check constraints

        Parameters:
        -----------
        xs: np.ndarray [n_candidate, n_input]
            Input variables of the candidates.
        pre_processing_problem: Problem
            Problem for pre-processing.
        user_pre_processing_func: Callable|None
            User-defined function to evaluate the individuals.
            If None, use external evaluation script.

        Returns:
        --------
        feasibility_flags: List[bool] [n_candidate]
            Feasibility flags of the candidates.
        ID_list: List[int]
            List of IDs of the candidates.
        '''
        self.opt.log(f'Checking pre-processing feasibility of {xs.shape[0]} candidates...', level=2, prefix='  > ')

        pre_processing_problem.calculation_folder = os.path.join(
            self.opt.dir_save, self.pre_process_folder)

        db = Database(pre_processing_problem, database_type='total')
        for i in range(xs.shape[0]):
            indi = Individual(pre_processing_problem, x=xs[i], ID=i+1)
            # Duplicate and bound checks are disabled on purpose: the returned
            # flags must line up one-to-one with the rows of `xs`, and
            # `_adjust_x_values_by_valid_database` rejects a length mismatch.
            added, warning_info = db.add_individual(
                indi, check_duplication=False, check_bounds=False,
                print_warning_info=False)
            if not added:
                self.opt.log(warning_info, level=2, prefix='  - ')


        # The local IDs restart at 1 on every call, so without a per-iteration
        # prefix the external check of iteration 2 would run in --- and read
        # back --- the folders of iteration 1.
        db.evaluate_individuals(mp_evaluation=self.opt.mp_evaluation,
                                user_func=user_pre_processing_func,
                                prefix_folder_name=f'iter{self.opt.iteration}-')

        feasibility_flags = []
        ID_list = []
        for indi in db.individuals:

            is_feasible = indi.valid_evaluation and indi.sum_violation <= 0.0
            feasibility_flags.append(is_feasible)
            ID_list.append(indi.ID)

            if not is_feasible:
                self.opt.log(f'Candidate #{indi.ID:2d} is infeasible.', level=2, prefix='  - ')

        return feasibility_flags, ID_list

    def _adjust_x_values_by_valid_database(self, xs: np.ndarray,
                    feasibility_flags: List[bool],
                    min_scaled_distance: float = 0.01,
                    max_scaled_distance: float = 0.10,
                    ID_list: List[int]|None = None) -> np.ndarray:
        '''
        Adjust the input variables of candidates towards the valid individuals in `db_valid`.

        Parameters:
        -----------
        xs: np.ndarray [n_candidate, n_input]
            Input variables of the candidates.
        feasibility_flags: List[bool] [n_candidate]
            Feasibility flags of the candidates.
        min_scaled_distance: float
            Minimum scaled distance to the valid individuals in `db_valid`.
        max_scaled_distance: float
            Maximum scaled distance to the valid individuals in `db_valid`.
        ID_list: List[int]
            List of local IDs of the candidates.
            If None, use index of `xs` as the list of IDs.

        Returns:
        --------
        xs_new: np.ndarray [n_candidate, n_input]
            Input variables of the candidates after adjustment.
        '''
        self.opt.log('Adjusting candidates based on the valid database...', level=2, prefix='  > ')

        xs_new = xs.copy()

        # Keep compatibility with list/ndarray inputs, but fail fast when
        # upstream feasibility checks return inconsistent lengths.
        feasibility_flags_arr = np.asarray(feasibility_flags, dtype=bool).reshape(-1)
        n_candidate = xs_new.shape[0]
        if feasibility_flags_arr.size != n_candidate:
            raise ValueError(
                f"Length mismatch: feasibility_flags has {feasibility_flags_arr.size} "
                f"entries, but xs has {n_candidate} candidates."
            )

        index_infeasible = np.where(~feasibility_flags_arr)[0]

        if ID_list is not None:
            ID_list_infeasible = [ID_list[i] for i in index_infeasible]
        else:
            ID_list_infeasible = None

        xs_adjusting = xs_new[index_infeasible]

        xs_adjusting = self._restrict_x_values_by_valid_database(xs_adjusting,
                                min_scaled_distance=min_scaled_distance,
                                max_scaled_distance=max_scaled_distance,
                                ID_list=ID_list_infeasible)

        xs_new[index_infeasible] = xs_adjusting

        return xs_new


class PostProcess(ABC):
    '''
    Post-processing of `db_total` databases in each iteration.

    The databases are accessed through the `OptBaseFramework` object, `opt`.
    The `db_total` databases are modified in place.

    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object.
    '''
    def __init__(self, opt: OptBaseFramework):

        self.opt = opt

    @abstractmethod
    def apply(self) -> None:
        '''
        Apply the post-processing to the `db_total` databases.
        '''
        self.opt.log(f'Post-processing of {self.opt.db_total.size} individuals in the total database started.', level=1)
        pass


