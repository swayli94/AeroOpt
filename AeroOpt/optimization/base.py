'''
Base framework for optimization.
'''
import os
import numpy as np
import time

from AeroOpt.core import (
    Problem, Database,
    SettingsOptimization,
    init_log, log
)
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
    pre_process: PreProcess
        Pre-processing of the `db_candidate` database to be evaluated.
    post_process: PostProcess
        Post-processing of the `db_candidate` database after evaluation.
    '''
    def __init__(self, problem: Problem,
            optimization_settings: SettingsOptimization,
            pre_process: 'PreProcess' = None,
            post_process: 'PostProcess' = None):
        
        self.problem = problem
        self.optimization_settings = optimization_settings

        self.pre_process : 'PreProcess' = pre_process
        self.post_process : 'PostProcess' = post_process
        
        self.iteration : int = 0
        
        # Database
        self.db_total = Database(self.problem, database_type='total')
        self.db_valid = Database(self.problem, database_type='valid')
        self.db_elite = Database(self.problem, database_type='elite')
        self.db_candidate = Database(self.problem, database_type='population')
        
        # Analysis of the database
        self.analyze_total = AnalyzeDatabase(self.db_total,
                               variables_for_calculating_potential=None,
                               critical_potential=self.optimization_settings.critical_potential_x)
        
        init_log(self.dir_summary, self.fname_log)
        
        self.log(f'Optimization [{self.name}] initialized.', level=0)
        
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
        return self.optimization_settings.name + self.problem.name
    
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
        return np.max(self.db_total._id_list)

    #* Main procedures
    
    def main(self) -> None:
        '''
        Main loop of the optimization.
        '''
        self.resume()
        
        self.initialize_population()
        
        self.select_valid_elite_from_total()
        
        while not self.termination():
            
            self.iteration += 1
            t0 = time.perf_counter()
            self.log(f'Iteration {self.iteration} started.', level=1)
            
            self.update_parameters()
            
            self.generate_candidate_individuals()
            
            if self.pre_process is not None:
                self.pre_process.apply()
                
            self.evaluate_db_candidate()
            
            if self.post_process is not None:
                self.post_process.apply()
                
            self.select_valid_elite_from_total()
            
            t1 = time.perf_counter()
            self.log(f'Iteration {self.iteration} finished in {(t1-t0)/60.0:.2f} min.', level=1)
    
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
        Initialize the initial population database `db_candidate`,
        including:
        - generation (Design of Experiments, perturbation, user-defined, etc.)
        - pre-processing
        - evaluation
        - post-processing
        '''
        self.iteration = 1
        raise NotImplementedError('Not implemented.')

    def termination(self) -> bool:
        '''
        Check if the optimization should be terminated.
        '''
        return self.iteration >= self.max_iterations

    def update_parameters(self) -> None:
        '''
        Update settings and parameters of the optimization.
        '''
        return None
    
    def generate_candidate_individuals(self) -> None:
        '''
        Generate `db_candidate` database from `db_valid` database,
        including:
        - selection of the parent database
        - evolution (crossover, mutation, etc.) of the parent database
        - user-defined new individuals
        - search from surrogate models
        - assignment of new IDs
        '''
        raise NotImplementedError('Not implemented.')

    def select_valid_elite_from_total(self) -> None:
        '''
        Select valid and elite individuals from the total database.
        '''
        raise NotImplementedError('Not implemented.')

    def evaluate_db_candidate(self) -> None:
        '''
        Evaluate the `db_candidate` database,
        then add the individuals to `db_total`.
        '''
        t0 = time.perf_counter()
        
        self.db_candidate.evaluate_individuals()
        
        self.db_total.merge_with_database(self.db_candidate, deepcopy=True)

        t1 = time.perf_counter()
        self.log(f'Evaluation finished in {(t1-t0)/60.0:.2f} min.', level=1)
    
    #* Support functions

    def log(self, text: str, level: int = 1) -> None:
        '''
        Log a message to the log file.
        '''
        log(text, prefix='>>> ', fname=self.fname_log,
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
        
    def apply(self) -> None:
        '''
        Apply the pre-processing to the `db_candidate` database.
        '''
        raise NotImplementedError('Pre-processing is not implemented.')
    
    def _restrict_x_values(self, xs: np.ndarray,
                        min_scaled_distance: float = 0.0,
                        max_scaled_distance: float = 1.0,
                        ) -> np.ndarray:
        '''
        Restrict the input variables of candidates,
        so that their scaled distances to the existed individuals in `db_total`
        are within [min_scaled_distance, max_scaled_distance].
        
        Parameters:
        -----------
        xs: np.ndarray [n_candidate, n_input]
            Input variables of the candidates.
        min_scaled_distance: float
            Minimum scaled distance to the existed individuals in `db_total`.
        max_scaled_distance: float
            Maximum scaled distance to the existed individuals in `db_total`.
        
        Returns:
        --------
        xs_new: np.ndarray [n_candidate, n_input]
            Input variables of the candidates after restriction.
        '''
        xs_new = np.zeros_like(xs)
        n_candidate = xs.shape[0]
        
        scaled_xs = self.opt.problem.scale_x(xs)
        
        distance_matrix = self.opt.analyze_total.calculate_distance_to_database(
                                scaled_xs, update_attributes=True) # [n_candidate, n_total]

        min_distance = np.min(distance_matrix, axis=1) # [n_candidate]
        
        critical_scaled_distance = self.opt.problem.critical_scaled_distance
        min_scaled_distance = max(min_scaled_distance, critical_scaled_distance)
        max_scaled_distance = max(max_scaled_distance, critical_scaled_distance)

        for i in range(n_candidate):
            
            if min_distance[i] < min_scaled_distance:
                
                j_total = np.argmin(distance_matrix[i])
                x_ref = self.opt.analyze_total.database.individuals[j_total].x
                dx = (xs[i] - x_ref)/(min_distance[i]+1e-6)
                xs_new[i] = x_ref + min_scaled_distance * dx
                
            elif min_distance[i] > max_scaled_distance:
                
                j_total = np.argmin(distance_matrix[i])
                x_ref = self.opt.analyze_total.database.individuals[j_total].x
                dx = (xs[i] - x_ref)/(min_distance[i]+1e-6)
                xs_new[i] = x_ref + max_scaled_distance * dx
                
            else:
                
                xs_new[i] = xs[i]
        
        xs_new = self.opt.problem.scale_x(xs_new, reverse=True)
        self.opt.problem.apply_bounds_x(xs_new)
                
        return xs_new


class PostProcess(object):
    '''
    Post-processing of `db_candidate` database in each iteration.
    
    The databases are accessed through the `OptBaseFramework` object, `opt`.
    The `db_candidate` is modified in place.
    
    Parameters:
    -----------
    opt: OptBaseFramework
        Optimization base framework object.
    '''
    def __init__(self, opt: OptBaseFramework):

        self.opt = opt

    def apply(self) -> None:
        '''
        Apply the post-processing to the `db_candidate` database.
        '''
        raise NotImplementedError('Post-processing is not implemented.')


