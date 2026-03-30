'''
Analyze and manipulate the database.
'''

import numpy as np
from typing import Dict, Tuple, List
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from aeroopt.core.problem import Problem
from aeroopt.core.database import Database
from aeroopt.core.individual import Individual

from aeroopt.analysis.utils import (
    calculate_potential_coefficient,
    func_potential
)


class AnalyzeDatabase(object):
    '''
    Analyze and manipulate the database.
    
    Parameters:
    -----------
    database: Database
        The database is not copied, so it can be modified in place.
        It can also be automatically updated when the database is modified outside.
    variables_for_calculating_potential: List[str]
        Variables to be used for calculating the potential.
        If None, all `x` variables will be used.
    critical_potential: float
        Critical potential to determine the typical distance (`d_typical`).
        Potential `phi` drops to `critical_potential` when `d=d_typical`
        A smaller `critical_potential` makes the potential lower.
        
    Attributes:
    -----------
    problem: Problem
        Problem of the database.
    size: int
        Size of the database.
    n_variable: int
        Number of variables (`v`) for calculating the potential.
        By default, all `x` variables will be used as `v`.
    statistics: Dict[str, np.ndarray]
        Statistics of all individuals.
    '''
    def __init__(self, database: Database,
                variables_for_calculating_potential: List[str] = None,
                critical_potential: float = 0.2):
        
        self.database = database
        self.critical_potential = critical_potential
        self.variables_for_calculating_potential = variables_for_calculating_potential
        
        self.statistics : Dict[str, np.ndarray] = None
        self._xs : np.ndarray = None
        self._ys : np.ndarray = None
        self._scaled_xs : np.ndarray = None
        self._scaled_ys : np.ndarray = None
        
        # Scaled variables for calculating the potential.
        self._scaled_variables : np.ndarray = None
        self._distance_matrix : np.ndarray = None
        self._coef_potential : float = None
        self._potentials : np.ndarray = None
        self._d_typical : float = None
        
        # Grouping by variables
        self._group_mean : np.ndarray = None # [n_group, n_variable]
        self._group_std : np.ndarray = None # [n_group, n_variable]
        self._ID_in_group : List[List[int]] = [] # [n_group][n_indi_in_group]

        # Settings
        self.metric : str = 'euclidean'
        
        self.update_attributes()
    
    @property
    def problem(self) -> Problem:
        return self.database.problem
    
    @property
    def size(self) -> int:
        return self.database.size
    
    @property
    def n_variable(self) -> int:
        '''
        Number of variables (`v`) for calculating the potential.
        
        By default, all `x` variables will be used.
        '''
        if self.variables_for_calculating_potential is None:
            return self.problem.n_input
        else:
            return len(self.variables_for_calculating_potential)
    
    @property
    def distance_matrix(self) -> np.ndarray:
        '''
        Scaled distance matrix of the database.
        - shape: [n, n]
        - d_ij is the distance of scaled variables `v` between the i-th and j-th individuals.
        - needs to be manually updated when the database is modified.
        '''
        return self._distance_matrix
    
    @property
    def coef_potential(self) -> float:
        '''
        Coefficient `c` for potential function `phi = (c*d+1)*exp(-c*d)`.
        - it is used to scale the distance `d` to the potential `phi`.
        - `phi=1` when `d=0`, and `phi=0` when `d` is large.
        - a larger `c` makes the potential decrease faster.
        '''
        return self._coef_potential
    
    @property
    def d_typical(self) -> float:
        '''
        Typical distance of the database.
        - it is the average minimum distance to adjacent points.
        - it is used to determine the `coef_potential`.
        - potential `phi` drops to `critical_potential` when `d=d_typical`.
        '''
        return self._d_typical
    
    @property
    def potentials(self) -> np.ndarray:
        '''
        Potential of each point in the database.
        - shape: [n]
        - potentials[i] is the potential of the i-th point
        - potentials[i] is the sum of potential due to all other points
        - it represents how much the i-th point is crowded by other points
        - it ranges from 0 to n-1
        '''
        return self._potentials
    
    @property
    def mean_potential(self) -> float:
        '''
        The average potential of all points in the database.
        '''
        return np.mean(self._potentials)
    
    @property
    def ID_in_group(self) -> List[List[int]]:
        '''
        IDs of the individuals in each group.
        '''
        return self._ID_in_group
    
    @property
    def n_group(self) -> int:
        '''
        Number of groups.
        '''
        return len(self._ID_in_group)
    
    #* Analysis of the database
    
    def assemble_variables_for_potential(self, xs: np.ndarray, ys: np.ndarray,
                        is_scaled_x: bool = False,
                        is_scaled_y: bool = False) -> np.ndarray:
        '''
        Assemble scaled variables for calculating the potential.
        The variables are copied to avoid modifying the original arrays.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Input variables
        ys: np.ndarray [n, n_output]
            Output variables
        is_scaled_x: bool
            If True, the input variables are already scaled.
        is_scaled_y: bool
            If True, the output variables are already scaled.
            
        Returns:
        --------
        scaled_variables: np.ndarray [n, n_variable]
            Scaled variables for calculating the potential.
        '''
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        if ys.ndim == 1:
            ys = ys[np.newaxis, :]
        
        if not is_scaled_x:
            xs = self.problem.scale_x(xs)
        else:
            xs = xs.copy()
        
        if not is_scaled_y:
            ys = self.problem.scale_y(ys)
        else:
            ys = ys.copy()
        
        if self.variables_for_calculating_potential is None:
            scaled_variables = xs
        else:
            
            n = xs.shape[0]
            n_var = len(self.variables_for_calculating_potential)
            scaled_variables = np.zeros([n, n_var])
            
            for i, var in enumerate(self.variables_for_calculating_potential):
                
                if var in self.problem.data_settings.name_input:
                    j = self.problem.data_settings.name_input.index(var)
                    scaled_variables[:, i] = xs[:, j]
                elif var in self.problem.data_settings.name_output:
                    j = self.problem.data_settings.name_output.index(var)
                    scaled_variables[:, i] = ys[:, j]
                else:
                    raise ValueError(f'Variable {var} is not in the problem.')
        
        return scaled_variables
    
    def update_data_arrays(self) -> None:
        '''
        Update the data arrays of the database, including:
        - xs, ys
        - scaled_xs, scaled_ys
        - scaled_variables
        '''
        nn = self.size
        self._xs = np.zeros([nn, self.problem.n_input])
        self._ys = np.zeros([nn, self.problem.n_output])
        self._scaled_xs = np.zeros([nn, self.problem.n_input])
        self._scaled_ys = np.zeros([nn, self.problem.n_output])
        
        for i in range(nn):
            self._xs[i,:] = self.database.individuals[i].x
            self._ys[i,:] = self.database.individuals[i].y
            
        self._scaled_xs = self.problem.scale_x(self._xs)
        self._scaled_ys = self.problem.scale_y(self._ys)
        
        self._scaled_variables = self.assemble_variables_for_potential(
            self._scaled_xs, self._scaled_ys, is_scaled_x=True, is_scaled_y=True)

    def update_statistics(self) -> Dict[str, np.ndarray]:
        '''
        Update the statistics of all individuals, including:
        average, standard deviation, minimum, maximum of (scaled) `x`,  `y` and `v`.
        '''
        if self.size <= 0:
            self.statistics = None
            return self.statistics

        self.statistics = {
            'average_x': np.mean(self._xs, axis=0),
            'average_y': np.mean(self._ys, axis=0),
            'std_x': np.std(self._xs, axis=0),
            'std_y': np.std(self._ys, axis=0),
            'min_x': np.min(self._xs, axis=0),
            'min_y': np.min(self._ys, axis=0),
            'max_x': np.max(self._xs, axis=0),
            'max_y': np.max(self._ys, axis=0),
            
            'average_scaled_x': np.mean(self._scaled_xs, axis=0),
            'average_scaled_y': np.mean(self._scaled_ys, axis=0),
            'std_scaled_x': np.std(self._scaled_xs, axis=0),
            'std_scaled_y': np.std(self._scaled_ys, axis=0),
            'min_scaled_x': np.min(self._scaled_xs, axis=0),
            'min_scaled_y': np.min(self._scaled_ys, axis=0),
            'max_scaled_x': np.max(self._scaled_xs, axis=0),
            'max_scaled_y': np.max(self._scaled_ys, axis=0),
            
            'average_v': np.mean(self._scaled_variables, axis=0),
            'std_v': np.std(self._scaled_variables, axis=0),
            'min_v': np.min(self._scaled_variables, axis=0),
            'max_v': np.max(self._scaled_variables, axis=0),
        }
        
        return self.statistics

    def update_distance_matrix(self) -> np.ndarray:
        '''
        Update the distance matrix of the database.
        '''
        if self.size <= 0:
            self._distance_matrix = None
        else:
            self._distance_matrix = cdist(
                self._scaled_variables, self._scaled_variables, metric=self.metric)
        
        return self._distance_matrix

    def update_attributes(self) -> None:
        '''
        Update the attributes of the database.
        '''
        self.update_data_arrays()
        self.update_statistics()
        self.update_distance_matrix()

    def calculate_typical_distance(self,
                update_attributes: bool = True,
                ) -> float:
        '''
        Calculate the typical distance of the database,
        and the coefficient for potential function.
        
        Parameters:
        -----------
        update_attributes: bool
            If True, the data arrays, statistics, and distance matrix will be updated.

        Returns:
        --------
        d_typical: float
            Average minimum distance to adjacent points.
            The distance to neighbor when the potential is equal to `critical_potential`.
        '''
        nn = self.size
        if nn <= 1:
            raise ValueError('The database must have at least 2 individuals.')
        
        if update_attributes:
            self.update_attributes()
        
        # Typical distance is the average nearest-neighbor distance.
        dm = np.array(self._distance_matrix, copy=True)
        np.fill_diagonal(dm, np.inf)
        ds = np.min(dm, axis=1) # [nn]
        self._d_typical = float(np.mean(ds))
        
        if self._d_typical <= 0.0:
            self._d_typical = self.problem.critical_scaled_distance

        # Calculate coefficient for potential function.
        self._coef_potential = calculate_potential_coefficient(
            self._d_typical, self.critical_potential)
        
        # Assign `crowding_distance` of all individuals.
        for i in range(nn):
            self.database.individuals[i].crowding_distance = ds[i]

        return self._d_typical

    def calculate_crowding_metrics(self,
                update_attributes: bool = True,
                ) -> Tuple[float, np.ndarray]:
        '''
        Calculate the crowding metrics of the database, including:
        - Average minimum distance to adjacent points
        - Potential of each point
        
        Parameters:
        -----------
        update_attributes: bool
            If True, the data arrays, statistics, and distance matrix will be updated.

        Returns:
        --------
        d_typical: float
            Average minimum distance to adjacent points.
            The distance to neighbor when the potential is equal to `critical_potential`.
        potentials: np.ndarray
            Potential of each point.
        '''
        nn = self.size
        
        self.calculate_typical_distance(update_attributes=update_attributes)

        self._potentials = func_potential(
            self._distance_matrix, self._coef_potential) # [nn, nn]
        
        self._potentials = np.sum(self._potentials, axis=1) - 1.0 # [nn]
        
        # Assign `crowding_potential` of all individuals.
        for i in range(nn):
            self.database.individuals[i].crowding_potential = self._potentials[i]

        return self._d_typical, self._potentials

    #* Evaluation of other individuals

    def calculate_distance_to_database(self,
                vs: np.ndarray,
                update_attributes: bool = True,
                ) -> np.ndarray:
        '''
        Calculate the scaled distance to the database.
        
        Parameters:
        -----------
        vs: np.ndarray [n, n_variable] or [n_variable]
            Scaled variables of the points to be evaluated.
        update_attributes: bool
            If True, the data arrays, statistics, and distance matrix will be updated.
        
        Returns:
        --------
        distance_matrix: np.ndarray [n, nn]
            Scaled distance to all points in the database.
        '''
        #nn = self.size
        if update_attributes:
            self.update_data_arrays()
        
        _vs = np.atleast_2d(vs)
        
        return cdist(_vs, self._scaled_variables, metric=self.metric)

    def calculate_potential_induced_by_database(self,
                vs: np.ndarray,
                update_attributes: bool = True,
                ) -> np.ndarray:
        '''
        Calculate the potential at `vs` induced by the database individuals.
        
        Parameters:
        -----------
        vs: np.ndarray [n, n_variable] or [n_variable]
            Scaled variables of the points to be evaluated.
        update_attributes: bool
            If True, the data arrays, statistics, and distance matrix will be updated.

        Returns:
        --------
        potentials: np.ndarray [n]
            Potential at each point induced by the database individuals.
        '''
        #nn = self.size

        if update_attributes:
            self.update_attributes()

        if self._coef_potential is None:
            self.calculate_crowding_metrics(update_attributes=False)

        _vs = np.atleast_2d(vs)
            
        # n = _vs.shape[0]
        distance_matrix = cdist(_vs, self._scaled_variables, metric=self.metric) # [n, nn]
        
        potentials = func_potential(
            distance_matrix, self._coef_potential) # [n, nn]
        potentials = np.sum(potentials, axis=1) # [n]
        
        if vs.ndim == 1:
            potentials = potentials[0]
        
        return potentials

    #* Manipulation of the database
    
    def eliminate_crowding_individuals(self,
                threshold_distance: float = 0.01,
                threshold_potential: float = 0.8,
                n_min_left: int = None,
                n_max_delete: int = None,
                ) -> List[int]:
        '''
        Eliminate individuals with low crowding distance or high crowding potential.

        This function updates crowding metrics (distance/potential) after each deletion,
        re-sorts the database, and then selects the next deletion candidate.
        
        Parameters:
        -----------
        threshold_distance: float
            Threshold distance to determine the individuals to be eliminated.
            Individuals with `crowding_distance < threshold_distance` will be eliminated.
            `crowding_distance` is the minimum distance to adjacent points.
        threshold_potential: float
            Threshold potential to determine the individuals to be eliminated.
            Individuals with `crowding_potential > threshold_potential` will be eliminated.
            `crowding_potential` is the potential induced by all other points.
        n_min_left: int
            Minimum number of individuals left in the database.
            If None, no limit is applied.
        n_max_delete: int
            Maximum number of individuals to be deleted.
            If None, no limit is applied.
            
        Returns:
        --------
        ID_list_eliminated: List[int]
            List of IDs of the individuals that are eliminated.
        '''
        if self.size <= 1:
            return []
        
        if n_min_left is not None:
            if self.size <= n_min_left:
                return []
        else:
            n_min_left = 1
            
        if n_max_delete is None:
            n_max_delete = self.size - n_min_left
        if n_max_delete <= 0:
            return []

        self.calculate_crowding_metrics(update_attributes=True)
            
        # Sort database by crowding metrics
        self.database.sort_database(sort_type=6)

        def _should_eliminate(indi: Individual) -> bool:
            # Keep individuals that are "well spaced and not too crowded":
            # - crowding_distance > threshold_distance
            # - crowding_potential < threshold_potential
            # Otherwise, eliminate.
            return not (
                indi.crowding_distance > threshold_distance
                and indi.crowding_potential < threshold_potential
            )

        ID_list_eliminated: List[int] = []
        need_final_update: bool = False

        # Delete one individual at a time. After each deletion, update
        # distance/potential and then re-sort before picking the next one.
        while len(ID_list_eliminated) < n_max_delete and self.size > n_min_left:

            candidate_index = None
            # Search from the end (worst crowding metrics first).
            for i in range(self.size - 1, -1, -1):
                if _should_eliminate(self.database.individuals[i]):
                    candidate_index = i
                    need_final_update = True
                    break

            # No more candidates.
            if candidate_index is None:
                need_final_update = True
                break

            candidate = self.database.individuals[candidate_index]
            ID_list_eliminated.append(candidate.ID)
            self.database.delete_individual(index=candidate_index)

            if self.size > 1:
                self.calculate_crowding_metrics(update_attributes=True)

            # Re-sort after each deletion so the next candidate is picked correctly.
            self.database.sort_database(sort_type=6)

        # If only 1 individual remains, crowding metrics cannot be computed.
        if self.size > 1 and need_final_update:
            self.calculate_crowding_metrics(update_attributes=True)

        return ID_list_eliminated

    #* Grouping of individuals

    def calculate_grouping(self,
                n_groups: int,
                update_attributes: bool = True,
                ) -> None:
        '''
        Calculate the grouping of the database using k-means algorithm.
        
        Parameters:
        -----------
        n_groups: int = 10
            Number of groups.
        update_attributes: bool = True
            If True, the data arrays, statistics, and distance matrix will be updated.
        '''
        if update_attributes:
            self.update_attributes()
        
        kmeans = KMeans(n_clusters=n_groups).fit(self._scaled_variables)
        self._ID_in_group = [[] for _ in range(n_groups)]
        
        # Assign group to each individual
        cluster_labels : np.ndarray = kmeans.labels_ # [n]
        for i, indi in enumerate(self.database.individuals):
            j = int(cluster_labels[i])
            indi.group = j
            self._ID_in_group[j].append(indi.ID)
            
        # Calculate mean and std of each group
        self._group_mean = np.zeros([n_groups, self.n_variable])
        self._group_std = np.zeros([n_groups, self.n_variable])
        for i in range(n_groups):
            self._group_mean[i,:] = np.mean(self._scaled_variables[cluster_labels == i], axis=0)
            self._group_std[i,:] = np.std(self._scaled_variables[cluster_labels == i], axis=0)
    
    def calculate_statistics_of_groups(self,
                name_variables: List[str] = None,
                ) -> Dict[str, np.ndarray]:
        '''
        Calculate the statistics of scaled variables in each group.
        
        Parameters:
        -----------
        name_variables: List[str]
            Names of the variables to be used for calculating the statistics.
            Default is all `x` and `y` variables.
        '''
        if self.n_group <= 0:
            raise ValueError('Grouping has not been calculated.')
        
        if name_variables is None:
            name_variables = self.problem.data_settings.name_input + \
                self.problem.data_settings.name_output

        self.database.update_id_list()
        
        n_var = len(name_variables)
        statistics = {
            'mean': np.zeros([self.n_group, n_var]),
            'std': np.zeros([self.n_group, n_var]),
            'min': np.zeros([self.n_group, n_var]),
            'max': np.zeros([self.n_group, n_var]),
        }
        
        scaled_variables = self.assemble_variables_for_potential(
            self._scaled_xs, self._scaled_ys, is_scaled_x=True, is_scaled_y=True)
        
        for i in range(self.n_group):
            
            index_in_group = []
            for j in self._ID_in_group[i]:
                index_in_group.append(self.database.get_index_from_ID(j))
                
            group_variables = scaled_variables[index_in_group]
            
            statistics['mean'][i,:] = np.mean(group_variables, axis=0)
            statistics['std'][i,:] = np.std(group_variables, axis=0)
            statistics['min'][i,:] = np.min(group_variables, axis=0)
            statistics['max'][i,:] = np.max(group_variables, axis=0)
        
        return statistics
    