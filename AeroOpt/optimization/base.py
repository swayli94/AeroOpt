'''
Base framework for optimization.
'''
import os
import numpy as np
import time

from typing import List, Callable, Tuple

from AeroOpt.core import (
    Problem, Individual, Database,
    MultiProcessEvaluation,
    init_log, log
)
from AeroOpt.optimization.settings import SettingsOptimization
from AeroOpt.analysis.analyze_database import AnalyzeDatabase


class OptBaseFramework(object):
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
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
        
    Attributes:
    -----------
    iteration: int
        The current iteration number.
    pre_process: PreProcess
        Pre-processing of the `db_candidate` database to be evaluated.
    post_process: PostProcess
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
            user_func: Callable = None,
            mp_evaluation: MultiProcessEvaluation = None):
        
        self.problem = problem
        self.optimization_settings = optimization_settings

        self.user_func : Callable = user_func
        self.mp_evaluation : MultiProcessEvaluation = mp_evaluation
        
        self.iteration : int = 0
        
        # Processing objects manually defined in the main program.
        self.pre_process : 'PreProcess' = None
        self.post_process : 'PostProcess' = None
        
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
        if self.db_total.size <= 0:
            return 0
        return int(np.max(self.db_total._id_list))

    #* Main procedures
    
    def main(self) -> None:
        '''
        Main loop of the optimization.
        '''
        self.resume()
        
        self.initialize_population()
        
        self.select_elite_from_valid()
        
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
                
            self.select_elite_from_valid()
            
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
            indi.source = 'prev_database'

        self.iteration = 0
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
        t0 = time.perf_counter()
        self.log(f'Initial population preparation started.', level=1)
        
        self.generate_initial_individuals()
        
        if self.pre_process is not None:
            self.pre_process.apply()
            
        self.evaluate_db_candidate()
        
        self.update_total_and_valid_with_candidate()
        
        if self.post_process is not None:
            self.post_process.apply()
            
        self.log(f"Initial population prepared: valid={self.db_valid.size}.", level=1)
        
    #TODO: Can be adapted
    def generate_initial_individuals(self) -> None:
        '''
        Generate the initial individuals for optimization.
        
        - this is the default implementation with random sampling.
        - can be adapted to other methods, e.g., Design of Experiments, perturbation, user-defined, etc.
        - the initial individuals are stored in `db_candidate` database.
        '''
        xs = np.random.rand(self.population_size, self.problem.n_input)
        xs = self.problem.scale_x(xs, reverse=True)
        
        self.db_candidate.empty_database()
        for x in xs:
            indi = Individual(problem=self.problem, x=x)
            indi.source = "random"
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
    
    #! Needs to be implemented
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
        raise NotImplementedError('Not implemented.')

    def evaluate_db_candidate(self) -> None:
        '''
        Evaluate the `db_candidate` database,
        then add the individuals to `db_total`.
        '''
        t0 = time.perf_counter()
        
        self.db_candidate.evaluate_individuals(mp_evaluation=self.mp_evaluation,
                                            user_func=self.user_func)

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
        
        self.db_total.merge_with_database(
            self.db_candidate, deepcopy=True, log_func=self.log)

        self.db_valid.copy_from_database(self.db_total, deepcopy=True)
        self.db_valid.eliminate_invalid_individuals()

        # `db_total` / `db_valid` objects are not reassigned here (in-place merge + copy_into),
        # but keep analyzers pinned to the canonical databases in case they were ever pointed
        # elsewhere (e.g. stale `db_candidate`, or legacy code that swapped `db_valid`).
        if self.analyze_total is not None:
            self.analyze_total.database = self.db_total
        if self.analyze_valid is not None:
            self.analyze_valid.database = self.db_valid
        
        n_added_total = self.db_total.size - n_previous_total
        n_added_valid = self.db_valid.size - n_previous_valid
        
        self.log(f'Add {n_added_total} individuals to total, updated to {self.db_total.size}.', level=1)
        self.log(f'Add {n_added_valid} individuals to valid, updated to {self.db_valid.size}.', level=1)

    #! Needs to be implemented
    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database:
        
        - Pareto-dominance ranking (e.g., NSGA-II)
        - Crowding-distance assignment (e.g., NSGA-II)
        - other selection methods (e.g., RVEA, MOEA/D, etc.)
        '''
        raise NotImplementedError('Not implemented.')

    #* Support functions

    def log(self, text: str, level: int = 1, prefix: str = '>>> ') -> None:
        '''
        Log a message to the log file.
        '''
        log(text, prefix=prefix, fname=self.fname_log,
                print_on_screen=(level<=self.level))

    def _assign_ID_to_candidate_individuals(self) -> None:
        '''
        Assign new IDs to the individuals in `db_candidate`.
        '''
        id_max = self.max_ID + 1
        for i in range(self.db_candidate.size):
            self.db_candidate.individuals[i].ID = id_max + i


class PreProcess(object):
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
        
    def apply(self) -> None:
        '''
        Apply the pre-processing to the `db_candidate` database.
        '''
        self.opt.log(f'Pre-processing of {self.opt.db_candidate.size} candidates started.', level=1)
        raise NotImplementedError('Pre-processing is not implemented.')
    
    def _restrict_x_values_by_valid_database(self, xs: np.ndarray,
                        min_scaled_distance: float = 0.0,
                        max_scaled_distance: float = 1.0,
                        ID_list: List[int] = None,
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
        ID_list: List[int]
            List of local IDs of the `xs` to be restricted.
            If None, use index of `xs` as the list of IDs.
        
        Returns:
        --------
        xs_new: np.ndarray [n_candidate, n_input]
            Input variables of the candidates after restriction.
        '''
        xs_new = np.zeros_like(xs)
        n_candidate = xs.shape[0]

        if n_candidate <= 0 or self.opt.db_valid.size <= 0:
            return xs_new
        
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
            
            min_d = max(min_distance[i], 1e-6)
            
            if min_d < min_scaled_distance:
                
                j_valid = np.argmin(distance_matrix[i])
                indi_ref = self.opt.analyze_valid.database.individuals[j_valid]
                ratio = min_scaled_distance / min_d
                xs_new[i] = indi_ref.x + ratio * (xs[i] - indi_ref.x)
                
                self.opt.log(f'Candidate #{ID_list[i]:2d}: too close to the' +
                            f' nearest valid individual X (ID={indi_ref.ID:4d}),' + 
                            f' adjust distance by ratio {ratio:.2f} away from X.',
                            level=2, prefix='  - ')
                
            elif min_d > max_scaled_distance:
                
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
        
        return xs_new

    def _check_pre_processing_feasibility(self, xs: np.ndarray,
                        pre_processing_problem: Problem,
                        user_pre_processing_func: Callable = None) -> Tuple[List[bool], List[int]]:
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
        user_pre_processing_func: Callable
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
            added, warning_info = db.add_individual(indi, print_warning_info=False)
            if not added:
                self.opt.log(warning_info, level=2, prefix='  - ')
            
        db.evaluate_individuals(mp_evaluation=self.opt.mp_evaluation,
                                user_func=user_pre_processing_func)
        
        feasibility_flags = []
        ID_list = []
        for indi in db.individuals:
            
            is_feasible = indi.valid_evaluation and indi.sum_violation <= 0.0
            feasibility_flags.append(is_feasible)
            ID_list.append(indi.ID)
            
            if not is_feasible:
                self.opt.log(f'Candidate #{indi.ID:2d} is infeasible.', level=2, prefix='  - ')
            
        return feasibility_flags, ID_list
    
    def _adjust_x_values_by_valid_database(self, xs: np.ndarray, feasibility_flags: List[bool],
                    min_scaled_distance: float = 0.01,
                    max_scaled_distance: float = 0.10,
                    ID_list: List[int] = None) -> np.ndarray:
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
        self.opt.log(f'Adjusting candidates based on the valid database...', level=2, prefix='  > ')
        
        xs_new = xs.copy()

        # Keep compatibility with list/ndarray inputs, but fail fast when
        # upstream feasibility checks return inconsistent lengths.
        feasibility_flags = np.asarray(feasibility_flags, dtype=bool).reshape(-1)
        n_candidate = xs_new.shape[0]
        if feasibility_flags.size != n_candidate:
            raise ValueError(
                f"Length mismatch: feasibility_flags has {feasibility_flags.size} "
                f"entries, but xs has {n_candidate} candidates."
            )
        
        index_infeasible = np.where(~feasibility_flags)[0]
        
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
        

class PostProcess(object):
    '''
    Post-processing of `db_total`, `db_valid` databases in each iteration.
    
    The databases are accessed through the `OptBaseFramework` object, `opt`.
    The `db_total` and `db_valid` databases are modified in place.
    
    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object.
    '''
    def __init__(self, opt: OptBaseFramework):

        self.opt = opt

    def apply(self) -> None:
        '''
        Apply the post-processing to the `db_total` and `db_valid` databases.
        '''
        raise NotImplementedError('Post-processing is not implemented.')


