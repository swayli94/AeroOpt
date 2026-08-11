'''
Individual definition.
'''

from __future__ import annotations

import copy

import numpy as np
from typing import Tuple, Dict, Any
from aeroopt.core.problem import Problem
from aeroopt.core.settings import SettingsData
from aeroopt.core.utils import compare_ndarray


ID_UNASSIGNED = -999

#* Sort types, see `SettingsProblem.sort_type_dict` and `Individual.__lt__`.
SORT_BY_DOMINANCE_AND_CROWDING = 0
SORT_BY_ID = 1
SORT_BY_X = 2
SORT_BY_Y = 3
SORT_BY_OBJECTIVES = 4
SORT_BY_DIVERSITY_OUTPUT = 5
SORT_BY_CROWDING = 6


class Individual:
    '''
    Individual of a problem.

    Parameters:
    -----------
    problem: Problem
        Problem of the individual.
    x: np.ndarray
        Input variables of the individual.
    ID: int
        ID of the individual (non-negative integer).
        The default value is -999, which means the ID is not assigned.
    y: np.ndarray
        Output variables of the individual.
    '''
    def __init__(self, problem: Problem, x: np.ndarray,
                    ID: int = ID_UNASSIGNED,
                    y: np.ndarray | None = None):

        self.problem = problem
        self.name_problem = problem.name

        self.x : np.ndarray = x
        self.ID : int = ID

        self.valid_evaluation : bool = True
        self.source : str = 'default'
        self.sort_type : int = SORT_BY_DOMINANCE_AND_CROWDING

        #* Output; an empty array means "not evaluated yet".
        if y is None:
            self.y : np.ndarray = np.array([])
        elif np.isscalar(y):
            self.y = np.array([y], dtype=float)
        else:
            self.y = np.asarray(y).copy()

        #* Scaled data
        self._scaled_x : np.ndarray = self.problem.scale_x(self.x)
        self._scaled_y : np.ndarray | None = (
            None if self.y.size == 0 else self.problem.scale_y(self.y))

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

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'Individual':
        '''
        Copy the individual's own data, but keep pointing at the same problem.

        Individuals are deep-copied in bulk --- `db_valid` is rebuilt from
        `db_total` on every iteration --- and the default recursion would clone
        the whole `Problem` behind each one: its settings arrays and its
        constraint callables, once per individual. That is most of the memory
        and time of an archive, it silently freezes each individual against the
        problem as it was when the copy was made, and it fails outright when a
        constraint callable holds something that cannot be copied.

        The problem describes the study, not the design, so it is shared.
        '''
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new

        for key, value in self.__dict__.items():
            if key == 'problem':
                new.problem = value
            else:
                setattr(new, key, copy.deepcopy(value, memo))

        return new

    @property
    def is_evaluated(self) -> bool:
        '''
        Whether this individual carries an output vector, i.e. `y` is not empty.
        '''
        return self.y.size > 0

    def update_x(self, x: np.ndarray) -> None:
        '''
        Replace the input vector and refresh the cached scaled input.

        `scaled_x` is cached at construction and drives duplication checks,
        crowding and every distance in the analysis layer. Assigning `x`
        directly leaves that cache describing the previous design.

        Parameters:
        -----------
        x: np.ndarray [n_input]
            New input vector.
        '''
        self.x = np.asarray(x, dtype=float)
        self._scaled_x = self.problem.scale_x(self.x)

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

        if self.sort_type == SORT_BY_ID:

            return self.ID < other.ID

        if self.sort_type == SORT_BY_X:

            return compare_ndarray(self.x, other.x) == -1

        if self.sort_type == SORT_BY_Y:

            return compare_ndarray(self.y, other.y) == -1

        if self.sort_type == SORT_BY_OBJECTIVES:

            y1 = self.problem.get_output_by_type(self.y, [1, -1])
            y2 = self.problem.get_output_by_type(other.y, [1, -1])
            return compare_ndarray(y1, y2) == -1

        if self.sort_type == SORT_BY_DIVERSITY_OUTPUT:

            y1 = self.problem.get_output_by_type(self.y, [2])
            y2 = self.problem.get_output_by_type(other.y, [2])
            return compare_ndarray(y1, y2) == -1

        if self.sort_type == SORT_BY_CROWDING:

            if self.crowding_distance > other.crowding_distance:
                return True
            if self.crowding_potential < other.crowding_potential:
                return True
            return False

        #* When both are invalid individuals
        #* Sort by constraint violation (smaller is better)
        if self.sum_violation > 0.0 and other.sum_violation > 0.0:
            return self.sum_violation < other.sum_violation

        #* Otherwise, sort by dominance and crowding distance
        if self.pareto_rank < other.pareto_rank:
            return True
        if self.pareto_rank > other.pareto_rank:
            return False
        if self.crowding_distance > other.crowding_distance:
            return True
        if self.crowding_potential < other.crowding_potential:
            return True
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
        if not self.is_evaluated:
            return obj

        k = 0
        for i in range(self.problem.n_output):
            if abs(self.problem.output_type[i]) == 1:
                obj[k] = self.y[i]
                k += 1
        return obj

    @property
    def data(self) -> Dict[str, Any]:
        '''
        Data of this individual,
        ndarray is converted to list for JSON serialization.
        '''
        if self.constraint_violations is not None:
            constraint_violations = self.constraint_violations.tolist()
        else:
            constraint_violations = None

        data = {
            'ID': self.ID,
            'name_problem': self.name_problem,
            'x': self.x.tolist(),
            'y': self.y.tolist(),
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

        Returns zeros when the individual has not been evaluated.
        '''
        if not self.is_evaluated:
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

        #* An individual without an output vector has no objectives to compare.
        #* A failed evaluation is always the worse option, and two failures are
        #* mutually non-dominated. Checking this first keeps a total database
        #* (which normally holds failed runs) safe to rank.
        self_evaluated = self.is_evaluated and self.valid_evaluation
        other_evaluated = other.is_evaluated and other.valid_evaluation

        if not self_evaluated or not other_evaluated:
            if self_evaluated:
                return 1
            if other_evaluated:
                return -1
            return 9

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


