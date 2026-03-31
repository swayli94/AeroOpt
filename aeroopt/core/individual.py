'''
Individual definition.
'''

import numpy as np
from typing import Tuple, Dict, Any
from aeroopt.core.problem import Problem
from aeroopt.core.settings import SettingsData
from aeroopt.core.utils import compare_ndarray


class Individual(object):
    '''
    Individual of a problem.
    
    Parameters:
    -----------
    problem: Problem
        Problem of the individual.
    x: np.ndarray
        Input variables of the individual.
    ID: int
        ID of the individual.
    y: np.ndarray
        Output variables of the individual.
    '''
    def __init__(self, problem: Problem, x: np.ndarray,
                    ID: int | None = None, y: np.ndarray | None = None):
        
        self.problem = problem
        self.name_problem = problem.name
        
        self.x : np.ndarray = x
        self.y : np.ndarray | None = y
        self.ID : int | None = ID

        self.valid_evaluation : bool = True
        self.source : str = 'default'
        self.sort_type : int = 0
        
        #* Scaled data
        self._scaled_x : np.ndarray = self.problem.scale_x(self.x)
        self._scaled_y : np.ndarray | None = None
        
        #* Constraints
        self.constraint_violations : np.ndarray | None = None
        self.sum_violation : float = 0.0
        
        #* Parameters for analysis
        self.group : int = 0
        
        # Crowding distance: minimum distance to adjacent points
        self.crowding_distance : float = 1.0 # higher the better
        
        # Crowding potential: potential induced by all other points
        self.crowding_potential : float = 0.0 # lower the better
        
        #* Parameters for evolutionary algorithms
        self.generation : int = 0
        self.pareto_rank : int = 0 # lower the better
        self.mutation_rate : float = 0.9
        self.crossover_rate : float = 0.9
        
        if y is not None:
            if isinstance(y, int) or isinstance(y,float):
                self.y = np.array([y])
            else:
                self.y = y.copy()
                
            self._scaled_y = self.problem.scale_y(self.y)

    def __repr__(self):
        return f"indi-{self.ID}"
    
    def __str__(self):
        return f"Individual (ID={self.ID}) of problem {self.problem.name}"
    
    def __lt__(self, other):
        '''
        User defined comparison operator [<].
        This enables sort() for individual list.
        '''
        if not isinstance(other, Individual):
            return NotImplemented

        if self.sort_type == 1:

            return self.ID < other.ID

        elif self.sort_type == 2:

            return compare_ndarray(self.x, other.x) == -1

        elif self.sort_type == 3:

            return compare_ndarray(self.y, other.y) == -1

        elif self.sort_type == 4:

            y1 = self.problem.get_output_by_type(self.y, [1, -1])
            y2 = self.problem.get_output_by_type(other.y, [1, -1])
            return compare_ndarray(y1, y2) == -1

        elif self.sort_type == 5:

            y1 = self.problem.get_output_by_type(self.y, [2])
            y2 = self.problem.get_output_by_type(other.y, [2])
            return compare_ndarray(y1, y2) == -1

        elif self.sort_type == 6:

            if self.crowding_distance > other.crowding_distance:
                return True
            elif self.crowding_potential < other.crowding_potential:
                return True
            else:
                return False 

        else:
            #* When both are invalid individuals
            #* Sort by constraint violation (smaller is better)
            if self.sum_violation > 0.0 and other.sum_violation > 0.0:
                return self.sum_violation < other.sum_violation

            #* Otherwise, sort by dominance and crowding distance
            if self.pareto_rank < other.pareto_rank:
                return True
            elif self.pareto_rank > other.pareto_rank:
                return False
            elif self.crowding_distance > other.crowding_distance:
                return True
            elif self.crowding_potential < other.crowding_potential:
                return True
            else:
                return False 

    @property
    def source2int(self) -> int:
        '''
        Return integer i representing the source of individual
        '''
        return SettingsData.data_source_dict[self.source]
    
    @staticmethod
    def int2source(i: int) -> str:
        '''
        Convert integer i to the source name of individual
        '''
        name = 'unknown'
        for key, value in SettingsData.data_source_dict.items():
            if value == i:
                name = key
        return name

    @property
    def objectives(self) -> np.ndarray:
        '''
        Objectives of this individual, ndarray [n_objective]
        '''
        obj = np.zeros(self.problem.n_objective)
        k = 0
        for i in range(self.problem.n_output):
            if abs(self.problem.output_type[i]) == 1:
                obj[k] = self.y[i] if self.y is not None else 0.0
                k += 1
        return obj

    @property
    def data(self) -> Dict[str, Any]:
        '''
        Data of this individual,
        ndarray is converted to list for JSON serialization.
        '''
        if self.y is not None:
            y = self.y.tolist()
        else:
            y = None
            
        if self.constraint_violations is not None:
            constraint_violations = self.constraint_violations.tolist()
        else:
            constraint_violations = None
        
        data = {
            'ID': self.ID,
            'name_problem': self.name_problem,
            'x': self.x.tolist(),
            'y': y,
            'valid_evaluation': self.valid_evaluation,
            'source': self.source,
            'sort_type': self.sort_type,
            'constraint_violations': constraint_violations,
            'sum_violation': self.sum_violation,
            'group': self.group,
            'generation': self.generation,
            'crowding_distance': self.crowding_distance,
            'crowding_potential': self.crowding_potential,
            'pareto_rank': self.pareto_rank,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
        }
        return data

    @property
    def scaled_x(self) -> np.ndarray:
        '''
        Scaled input variables of this individual.
        '''
        return self._scaled_x
    
    @property
    def scaled_y(self) -> np.ndarray:
        '''
        Scaled output variables of this individual.
        '''
        if self.y is None:
            return np.zeros(self.problem.n_output, dtype=float)
        if self._scaled_y is None:
            self._scaled_y = self.problem.scale_y(self.y)
        return self._scaled_y

    def eval_constraints(self,
                use_another_problem: Problem | None = None) -> Tuple[float, np.ndarray]:
        '''
        Evaluate constraints of this individual.
        
        Parameters
        -------------
        use_another_problem: Problem
            Another problem to evaluate constraints.
            If None, use the problem of this individual.
        
        Returns
        -------------
        sum_violation: float
            Sum of the constraint violations, only the violated constraints are counted.
        constraint_violations: np.ndarray
            Violation values of all the constraints for this individual.
        '''
        if use_another_problem is None:
            self.sum_violation, self.constraint_violations = self.problem.eval_constraints(self.x, self.y)
        else:
            self.sum_violation, self.constraint_violations = use_another_problem.eval_constraints(self.x, self.y)
            
        return self.sum_violation, self.constraint_violations

    def check_dominance(self, other) -> int:
        '''
        Check Pareto dominance.
        
        Parameters
        -------------
        other: Individual
            Another individual to compare dominance.
        
        Returns
        -------------
        i_dominance: int
            Dominance relationship between self and other.
            - `0`: equal
            - `1`: self dominates other
            - `-1`: self is dominated by other
            - `9`: self and other are non-dominated
        '''
        if not isinstance(other, Individual):
            raise ValueError(f'Must compare individuals, got {type(other)}')
        
        if other.problem != self.problem:
            raise ValueError(f'Must compare individuals of the same problem, got {self.problem.name} and {other.problem.name}')
        
        if self.constraint_violations is None:
            self.eval_constraints()
        if other.constraint_violations is None:
            other.eval_constraints()

        if self.sum_violation <= 0.0 and other.sum_violation > 0.0: 
            i_dominance = 1

        elif self.sum_violation > 0.0 and other.sum_violation <= 0.0: 
            i_dominance = -1

        elif self.sum_violation > 0.0 and other.sum_violation > 0.0: 
            i_dominance = 9

        else:
            i_dominance = self.problem.check_pareto_dominance(self.y, other.y)

        return i_dominance


