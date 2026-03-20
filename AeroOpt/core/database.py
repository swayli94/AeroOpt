'''
Database definition.
'''

import numpy as np
import json
import copy

from typing import List, Tuple
from AeroOpt.core.individual import Individual
from AeroOpt.core.problem import Problem


class Database(object):
    '''
    Basic database class.
    
    Parameters:
    -----------
    problem: Problem
        Problem of the database.
    database_type: str
        Type of the database.
    '''
    
    database_type_dict = {
        'default':      'not specified',
        'elite':        'elite database',
        'valid':        'valid database',
        'total':        'total database',
        'population':   'population database',
        'surrogate':    'surrogate model',
        'diversity':    'diversity calculation',
        'intersection': 'intersection of two databases',
        'sub-database': 'sub-database',
    }
    
    def __init__(self, problem: Problem, database_type='default'):

        self.problem = problem

        if database_type not in self.database_type_dict:
            raise ValueError(f'Invalid database type: {database_type}')
        self.database_type = database_type
        
        self.individuals : List[Individual] = []
        
        self._id_list : List[int] = [] # List of IDs in the order of individuals
        self._sorted : bool = False
    
    #* Attributes
    
    @property
    def size(self) -> int:
        '''
        Size of the database.
        '''
        return len(self.individuals)
    
    @property
    def sorted(self) -> bool:
        '''
        Whether the database is sorted.
        '''
        return self._sorted
    
    @property
    def critical_scaled_distance(self) -> float:
        '''
        Critical scaled distance for checking duplication of individuals.
        '''
        return self.problem.data_settings.critical_scaled_distance
        
    #* Basic functions
    
    def update_id_list(self) -> None:
        '''
        Update the list of IDs in the order of individuals.
        '''
        self._id_list = [indi.ID for indi in self.individuals]
        
    def sort_database(self, sort_type: int = 0) -> None:
        '''
        Sort the database by a certain type.
        
        Parameters:
        -----------
        sort_type: int
            Sort type defined in SettingsProblem.sort_type_dict.
            - 0: default, by dominance, crowding distance and potential
            - 1: sorting ID
            - 2: sorting x
            - 3: sorting y
            - 4: sorting objectives
            - 5: sorting type-2 output
            - 6: sorting crowding metrics
        '''
        for indi in self.individuals:
            indi.sort_type = sort_type
        self.individuals.sort()
        self._sorted = True
        self.update_id_list()
    
    #* Access data of individuals
    
    def get_index_from_ID(self, ID: int) -> int:
        '''
        Get index of an individual from its ID.
        '''
        return self._id_list.index(ID)
    
    def get_ID_from_index(self, index: int) -> int:
        '''
        Get ID of an individual from its index.
        '''
        return self.individuals[index].ID
    
    def get_largest_ID(self) -> int:
        '''
        Get the largest ID in the database.
        '''
        if self.size <= 0:
            return 0
        return np.max(self._id_list)
    
    def get_xs(self, scale: bool = False,
                ID_list: List[int] = None, 
                index_list: List[int] = None) -> np.ndarray:
        '''
        Get input variables of individuals in the database.
        
        Parameters:
        -----------
        scale:      bool
            If True, return scaled input variables.
        ID_list:    List[int]
            List of IDs of individuals to be selected.
        index_list: List[int]
            List of index of individuals to be selected.
            
        Returns:
        --------
        xs: np.ndarray [n, n_input]
            Input variables of individuals in the database.
        '''
        if self.size <= 0:
            return None
    
        if isinstance(ID_list, list):
            nn = len(ID_list)
            xs = np.zeros([nn, self.problem.n_input])
            for i in range(nn):
                ii = self.get_index_from_ID(ID_list[i])
                xs[i,:] = self.individuals[ii].x

        elif isinstance(index_list, list):
            nn = len(index_list)
            xs = np.zeros([nn, self.problem.n_input])
            for i in range(nn):
                xs[i,:] = self.individuals[index_list[i]].x

        else:
            nn = self.size
            xs = np.zeros([nn, self.problem.n_input])
            for i in range(nn):
                xs[i,:] = self.individuals[i].x

        if scale:
            xs = self.problem.scale_x(xs)

        return xs
    
    def get_ys(self, scale: bool = False,
                type_list: List[int] = None,
                ID_list: List[int] = None,
                index_list: List[int] = None) -> np.ndarray:
        '''
        Get output variables of individuals in the database.
        '''
        if self.size <= 0:
            return None
    
        if isinstance(ID_list, list):
            nn = len(ID_list)
            ys = np.zeros([nn, self.problem.n_output])
            for i in range(nn):
                ii = self.get_index_from_ID(ID_list[i])
                ys[i,:] = self.individuals[ii].y
    
        elif isinstance(index_list, list):
            nn = len(index_list)
            ys = np.zeros([nn, self.problem.n_output])
            for i in range(nn):
                ys[i,:] = self.individuals[index_list[i]].y
    
        else:
            nn = self.size
            ys = np.zeros([nn, self.problem.n_output])
            for i in range(nn):
                ys[i,:] = self.individuals[i].y
    
        if scale:
            ys = self.problem.scale_y(ys)
    
        if not type_list is None:
            ys = self.problem.get_output_by_type(ys, type_list)
    
        return ys
    
    #* Individual-level manipulation
    
    def check_duplication(self, x: np.ndarray,
                    is_scaled_x: bool = False
                    ) -> Tuple[List[bool]|bool, List[int]|int]:
        '''
        Check if the individual is duplicated.
        
        Note that the duplication is defined as the scaled distance between two individuals is less than a threshold, i.e.,
        `self.critical_scaled_distance`.
        
        Parameters:
        -----------
        x: np.ndarray [n, n_input] or [n_input]
            Input variables of the individual.
        is_scaled_x: bool
            If True, the input x is already scaled.
            
        Returns:
        --------
        is_duplicated: List[bool] or bool
            List of booleans indicating if the individuals are duplicated.
        closest_index: List[int] or int
            List of indices of the closest individuals.
        '''
        is_multiple = (x.ndim == 1)
        if is_multiple:
            n = x.shape[0]
            is_duplicated = [False] * n
            closest_index = [None] * n
        else:
            n = 1
            is_duplicated = False
            closest_index = None
        
        if self.size<=0:
            return is_duplicated, closest_index
        
        if not is_scaled_x:
            x = self.problem.scale_x(x)
        
        scaled_distance_matrix = self.problem.calculate_scaled_distance(
            x, self.get_xs(is_scaled_x=True),
            is_scaled_x=True)
        
        min_dis = np.min(scaled_distance_matrix, axis=1) # [n]
        closest_index = np.argmin(scaled_distance_matrix, axis=1) # [n]
    
        if is_multiple:
            for i in range(n):
                is_duplicated[i] = (min_dis[i] < self.critical_scaled_distance)
        else:
            if min_dis[0] < self.critical_scaled_distance:
                is_duplicated = True
            closest_index = closest_index[0]

        return is_duplicated, closest_index

    def add_individual(self, indi: Individual,
                    check_duplication: bool = True,
                    check_bounds: bool = True,
                    deepcopy: bool = True,
                    ) -> bool:
        '''
        Add an individual to the database.
        
        Only the individual ID may be modified,
        other attributes are not modified.
        
        Parameters:
        -----------
        indi: Individual
            Individual to be added.
        check_duplication: bool
            If True, check if the individual is duplicated.
        check_bounds: bool
            If True, check if the individual is out of bounds.
        deepcopy: bool
            If True, the individual is copied.
        
        Returns:
        --------
        added: bool
            True if the individual is added, False otherwise.
        '''
        # Check problem
        if indi.problem != self.problem:
            raise ValueError('Individual problem does not match database problem')
        
        # Check bounds
        if check_bounds:
            if not self.problem.check_bounds_x(indi.x):
                print(f'>>> Failed to add individual: ID={indi.ID}')
                print(f'    x out of bounds')
                return False
        
        if deepcopy:
            indi = copy.deepcopy(indi)
        
        # Check duplication
        if self.size <= 0:
            self.individuals.append(indi)
        else:
            is_duplicated, closest_index = self.check_duplication(indi.x)
            if is_duplicated and check_duplication:
                print(f'>>> Failed to add individual: ID={indi.ID}')
                print(f'    duplicated with ID {closest_index} in database')
                return False
        
        # Assign ID
        original_ID = indi.ID
        if original_ID is not None:
            if indi.ID in self._id_list:
                indi.ID = self.get_largest_ID() + 1
                print(f'>>> Assigned new ID {indi.ID} to individual: ID={original_ID}')
        else:
            indi.ID = self.get_largest_ID() + 1
            print(f'>>> Assigned new ID {indi.ID} to individual: ID= None')
            
        # Add individual to database
        self.individuals.append(indi)
        self._id_list.append(indi.ID)
        self._sorted = False
            
        return True
    
    def delete_individual(self, ID: int = None, index: int = -1) -> None:
        '''
        Delete an individual from the database.
        
        Parameters:
        -----------
        ID: int
            ID of the individual to be deleted.
        index: int
            Index of the individual to be deleted.
        '''
        if self.size <= 0:
            raise Exception('Can not delete individual from empty database')
        
        if ID is None and index == -1:
            self.individuals.pop()
            self._id_list.pop()

        elif ID in self._id_list:
            ii = self.get_index_from_ID(ID)
            self.individuals.pop(ii)
            self._id_list.pop(ii)

        elif index>=0 and index<self.size:
            self.individuals.pop(index)
            self._id_list.pop(index)
        
        else:
            raise Exception('ID or index not valid (size=%d)'%(self.size), ID, index)
        
        self._sorted = False
    
    def eliminate_invalid_individuals(self) -> None:
        '''
        Eliminate invalid individuals from the database.
        
        Invalid individuals are defined as:
        - the individual has no valid evaluation.
        - the individual is out of bounds.
        - the individual has constraint violations.
        - the constraints are updated in this function.
        '''
        # Loop backwards to avoid index shifting issues
        for i in range(self.size-1, -1, -1):
            
            indi = self.individuals[i]
            
            if not indi.valid_evaluation:
                self.delete_individual(index=i)
                continue
            
            if not self.problem.check_bounds_x(indi.x):
                self.delete_individual(index=i)
                continue
            
            if indi.y is not None:
                indi.eval_constraints()
            
            if indi.sum_violation > 0.0:
                self.delete_individual(index=i)
                continue
            
        self.update_id_list()
        self._sorted = False
    
    #* Database-level manipulation
    
    def get_sub_database(self,
                ID_list: List[int] = None, 
                index_list: List[int] = None,
                deepcopy: bool = True) -> 'Database':
        '''
        Create a sub-database from the database.
        
        Parameters:
        -----------
        ID_list: List[int]
            List of IDs of individuals to be selected.
        index_list: List[int]
            List of index of individuals to be selected.
        deepcopy: bool
            If True, the individuals are copied.
        
        Returns:
        --------
        sub_database: Database
            Sub-database created from the database.
        '''
        if ID_list is not None and index_list is not None:
            raise ValueError('Only one of ID_list and index_list should be provided')

        sub_database = Database(self.problem, database_type='sub-database')

        if ID_list is not None:
            sub_database.individuals = [self.individuals[self.get_index_from_ID(id_)] for id_ in ID_list]
        
        if index_list is not None:
            sub_database.individuals = [self.individuals[idx] for idx in index_list]
        
        if deepcopy:
            sub_database.individuals = [copy.deepcopy(indi) for indi in sub_database.individuals]
        
        sub_database.update_id_list()
        sub_database._sorted = False
        return sub_database

    def get_intersection_with_database(self,
                        other: 'Database',
                        deepcopy: bool = True) -> 'Database':
        '''
        Get the intersection of the database with another database.
        
        The intersection is defined as:
        - the `x` and `y` of individuals are the same in both databases.
        - the intersection is a new database.
        - the intersection database is picked from this database.
        
        Parameters:
        -----------
        other: Database
            Another database to get the intersection with.
        deepcopy: bool
            If True, the individuals are copied.
        '''
        if other.problem is not self.problem:
            raise ValueError('Databases must share the same problem instance')

        def _same_xy(a: Individual, b: Individual) -> bool:
            if not np.array_equal(a.x, b.x):
                return False
            if a.y is None and b.y is None:
                return True
            if a.y is None or b.y is None:
                return False
            return np.array_equal(a.y, b.y)

        intersection = Database(self.problem, database_type='intersection')
        for indi in self.individuals:
            if any(_same_xy(indi, o) for o in other.individuals):
                intersection.individuals.append(indi)
        
        if deepcopy:
            intersection.individuals = [copy.deepcopy(indi) for indi in intersection.individuals]
        
        intersection.update_id_list()
        intersection._sorted = False
        return intersection

    def merge_with_database(self,
                        other: 'Database',
                        deepcopy: bool = True) -> None:
        '''
        Merge the database with another database.
        
        The merge is defined as:
        - the individuals are merged into this database.
        - the duplicated individuals are not added.
        
        Parameters:
        -----------
        other: Database
            Another database to merge with.
        deepcopy: bool
            If True, the individuals are copied.
        '''
        if other.problem is not self.problem:
            raise ValueError('Databases must share the same problem instance')

        if other.size <= 0:
            return

        for indi in other.individuals:
            self.add_individual(indi, deepcopy=deepcopy)

        self.update_id_list()
        self._sorted = False

    def create_database_of_sub_problem(self,
                    sub_problem: Problem) -> 'Database':
        '''
        Create a database of a sub-problem.
        
        The sub-problem is defined as:
        - its names of `x` and `y` are subset of the current problem.
        - the other parameters can be different from the current problem.
        
        Parameters:
        -----------
        sub_problem: Problem
            Sub-problem to create the database.
        '''
        # Check if the sub-problem is a subset of the current problem
        if not sub_problem.is_subset_of(self.problem):
            raise ValueError('Sub-problem is not a subset of the current problem')
        
        # Create a new database for the sub-problem
        sub_database = Database(sub_problem, database_type='sub-database')
        
        # Mapping of `x` and `y` to the sub-problem
        parent_in = self.problem.data_settings.name_input
        parent_out = self.problem.data_settings.name_output
        ix = [parent_in.index(n) for n in sub_problem.data_settings.name_input]
        iy = [parent_out.index(n) for n in sub_problem.data_settings.name_output]
        
        # Add individuals to the sub-database
        # Pick components by parent-problem order, arranged as sub-problem order
        for indi in self.individuals:
            
            indi = copy.deepcopy(indi)
            
            indi.problem = sub_problem
            indi.name_problem = sub_problem.name
            
            indi.x = indi.x[ix].copy()
            indi._scaled_x = sub_problem.scale_x(indi.x)
            
            if indi.y is not None:
                indi.y = indi.y[iy].copy()
                indi._scaled_y = sub_problem.scale_y(indi.y)
                
            indi.eval_constraints()
            
            sub_database.add_individual(indi, deepcopy=False)
            
        sub_database.update_id_list()
        sub_database._sorted = False
        
        sub_database.evaluate_constraints()
        
        return sub_database

    #* Input and output
    
    def output_database_json(self, fname: str):
        '''
        Output database to JSON file.
        '''
        database_data = {
            'database_type': self.database_type,
            'individuals': [indi.data for indi in self.individuals]
        }
        
        with open(fname, 'w') as f:
            json.dump(database_data, f, indent=4)

    def read_database_json(self, fname: str):
        '''
        Read database from JSON file.
        '''
        with open(fname, 'r') as f:
            database_data = json.load(f)
            
        self.database_type = database_data['database_type']
        all_individuals = database_data['individuals']
        
        self.individuals = []
        for indi_data in all_individuals:
            
            indi = Individual(self.problem, x=np.array(indi_data['x']))
            
            for key, value in indi_data.items():
                
                if key == 'y':
                    if value is not None:
                        value = np.array(value)
                elif key == 'constraint_violations':
                    if value is not None:
                        value = np.array(value)
                    
                indi.__setattr__(key, value)
                
            self.individuals.append(indi)
        
        self._id_list = [indi.ID for indi in self.individuals]
        self._sorted = False

    #* Evaluation
    
    def evaluate_individuals(self) -> None:
        '''
        Evaluate the individuals in the database.
        '''
