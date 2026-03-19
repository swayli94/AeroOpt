'''
Analyze and manipulate the database.
'''

import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import cdist

from AeroOpt.core.problem import Problem
from AeroOpt.core.database import Database

from AeroOpt.analysis.utils import (
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
    statistics: Dict[str, np.ndarray]
        Statistics of all individuals.
    '''
    def __init__(self, database: Database, critical_potential: float = 0.2):
        
        self.database = database
        self.critical_potential = critical_potential
        
        self.statistics : Dict[str, np.ndarray] = None
        self._xs : np.ndarray = None
        self._ys : np.ndarray = None
        self._scaled_xs : np.ndarray = None
        self._scaled_ys : np.ndarray = None
        
        self._distance_matrix : np.ndarray = None
        self._coef_potential : float = None
        self._potentials : np.ndarray = None
        self._d_typical : float = None
        
        self.update_attributes()
    
    @property
    def problem(self) -> Problem:
        return self.database.problem
    
    @property
    def size(self) -> int:
        return self.database.size
    
    @property
    def distance_matrix(self) -> np.ndarray:
        '''
        Scaled distance matrix of the database.
        - shape: [n, n]
        - d_ij is the distance of scaled x between the i-th and j-th individuals.
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
    
    #* Analysis of the database
    
    def update_data_arrays(self) -> None:
        '''
        Update the data arrays of the database, including:
        - xs, ys
        - scaled_xs, scaled_ys
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

    def update_statistics(self) -> Dict[str, np.ndarray]:
        '''
        Update the statistics of all individuals, including:
        average, standard deviation, minimum, maximum of (scaled) x and y
        '''
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
        }
        
        return self.statistics

    def update_distance_matrix(self) -> np.ndarray:
        '''
        Update the distance matrix of the database.
        '''
        self._distance_matrix = cdist(
            self._scaled_xs, self._scaled_xs, metric='euclidean')

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

    def calculate_potential_of_database(self,
                xs: np.ndarray,
                is_scaled_x: bool = False,
                update_attributes: bool = True,
                ) -> np.ndarray:
        '''
        Calculate the potential of `xs` caused by all points in the database.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input] or [n_input]
            Input variables of the points to be evaluated.
        is_scaled_x: bool
            If True, the xs is already scaled.
        update_attributes: bool
            If True, the data arrays, statistics, and distance matrix will be updated.

        Returns:
        --------
        potentials: np.ndarray [n]
            Potential of each point caused by the database.
        '''
        nn = self.size
        if nn <= 1:
            raise ValueError('The database must have at least 2 individuals.')
        
        if update_attributes:
            self.update_attributes()

        if self._coef_potential is None:
            self.calculate_crowding_metrics(update_attributes=False)

        _xs = np.atleast_2d(xs)
            
        if not is_scaled_x:
            _xs = self.problem.scale_x(_xs)
            
        # n = _xs.shape[0]
        distance_matrix = self.problem.calculate_scaled_distance(
            _xs, self._scaled_xs, is_scaled_x=True) # [n, nn]
        
        potentials = func_potential(
            distance_matrix, self._coef_potential) # [n, nn]
        potentials = np.sum(potentials, axis=1) # [n]
        
        if xs.ndim == 1:
            potentials = potentials[0]
        
        return potentials

