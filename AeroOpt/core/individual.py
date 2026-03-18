'''
Individual definition.
'''

import numpy as np
from typing import Tuple
from AeroOpt.core.problem import Problem
from AeroOpt.core.utils import compare_ndarray


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
                    ID: int = None, y: np.ndarray = None):
        
        self.problem = problem
        
        self.x : np.ndarray = x
        self.y : np.ndarray = y
        self.ID : int = ID
        self.gen : int = 0
        self.group : int = -1
        self.source : str = 'default'
        self.sort_type : int = 0
        
        #* Constraints
        self.constraint_violations : np.ndarray = None
        self.sum_violation : float = 0.0
        
        #* Parameters for evolutionary algorithms
        self.crowding_distance : float = 1.0 # higher the better
        self.crowding_potential : float = 0.0 # lower the better
        self.pareto_rank : int = 0 # lower the better
        self.mutation_rate : float = 0.9
        self.crossover_rate : float = 0.9
        
        if y is not None:
            if isinstance(y, int) or isinstance(y,float):
                self.y = np.array([y])
            else:
                self.y = y.copy()

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
        return Individual.source_dict[self.source]
    
    @staticmethod
    def int2source(i: int) -> str:
        '''
        Convert integer i to the source name of individual
        '''
        name = 'unknown'
        for key, value in Individual.source_dict.items():
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
                obj[k] = self.y[i]
                k += 1
        return obj

    def eval_constraints(self, use_another_problem: Problem = None) -> Tuple[float, np.ndarray]:
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


