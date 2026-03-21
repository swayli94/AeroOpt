"""
Base class and shared utilities for evolutionary optimization algorithms.
"""

from typing import List, Callable
import numpy as np

from AeroOpt.core import (
    Database, Problem, 
    SettingsOptimization, MultiProcessEvaluation
)
from AeroOpt.optimization.base import (
    OptBaseFramework, PreProcess, PostProcess
)


class EvolutionaryAlgorithm(object):
    '''
    Shared utilities for evolutionary algorithms.

    This class refactors the common logic from legacy `Evolution`:
    - Pareto non-dominated ranking
    - Crowding-distance assignment
    - Elite extraction and best-individual transfer
    - Population shrinking with reserve-ratio protection
    
    Parameters:
    -----------
    algorithm_name: str
        Name of the evolutionary algorithm.
    '''
    def __init__(self, algorithm_name: str = "default"):
        self.algorithm_name = algorithm_name

    def __str__(self) -> str:
        return self.algorithm_name
    
    def __repr__(self) -> str:
        return self.algorithm_name

    @staticmethod
    def _get_unified_objectives(db: Database) -> np.ndarray:
        '''
        Return objective matrix with unified minimization direction.
        '''
        if db.size <= 0:
            return np.zeros((0, 0), dtype=float)

        n_obj = db.problem.n_objective
        ys = np.zeros((db.size, n_obj), dtype=float)
        for i, indi in enumerate(db.individuals):
            ys[i, :] = indi.objectives

        i_obj = 0
        for out_type in db.problem.problem_settings.output_type:
            if abs(out_type) != 1:
                continue
            if out_type == -1:
                ys[:, i_obj] = -ys[:, i_obj]
            i_obj += 1
        return ys

    @staticmethod
    def fast_non_dominated_ranking(db: Database) -> None:
        '''
        Non-dominated ranking of NSGAII proposed by Deb et al.,
        see [Deb2002], O(n^2).

        Updates `indi.pareto_rank` in-place (1-based).
        
        Note:
        ------
        - non-dominated ranking does not have transitivity,
        e.g., A = B and B > C, does not imply A > C.
        - Individuals of rank i are not dominated by individuals of rank j (j>i), but not necessarily dominate all individuals of rank j.
        - This may happen: A (rank 1) and B (rank 1) are non-dominated,
        A (rank 1) and C (rank 2) are also non-dominated, 
        C is rank 2 because B dominates C.
        '''
        n_indi = db.size
        if n_indi <= 0:
            return

        dominating_ith = [0 for _ in range(n_indi)]
        ith_dominated = [[] for _ in range(n_indi)]
        fronts = [[] for _ in range(n_indi + 1)]

        for p in range(n_indi - 1):
            for q in range(p + 1, n_indi):
                i_dominance = db.individuals[p].check_dominance(db.individuals[q])
                if i_dominance == 1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif i_dominance == -1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        for i in range(n_indi):
            if dominating_ith[i] == 0:
                fronts[0].append(i)
                db.individuals[i].pareto_rank = 1

        i_front = 0
        while len(fronts[i_front]) != 0:
            i_front += 1
            for p in fronts[i_front - 1]:
                if p < len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] == 0:
                            fronts[i_front].append(q)
                            db.individuals[q].pareto_rank = i_front + 1

    @staticmethod
    def pareto_dominance(y: np.ndarray, y_other: np.ndarray) -> int:
        '''
        Pareto dominance on already direction-unified objective vectors.

        Parameters:
        -----------
        y, y_other: np.ndarray [n_output]
            direction-unified objective vectors.
            
        Note:
        ------
        - equal: objs are equal
        - dominate: at least one obj is better, and others are equal
        - dominated: at least one obj is worse, and others are equal
        - non-dominated: some objs are better, and some are worse

        Returns:
        --------
        i_dominance: int
            dominance relationship between y and y_other
            - `0`: equal
            - `1`: y dominates y_other
            - `-1`: y is dominated by y_other
            - `9`: non-dominated
        '''
        dominances = np.sign(y - y_other).tolist()

        if 1 in dominances and -1 in dominances:
            return 9
        if 1 in dominances and -1 not in dominances:
            return 1
        if 1 not in dominances and -1 in dominances:
            return -1
        return 0

    @staticmethod
    def faster_non_dominated_ranking(db: Database,
            is_valid_database: bool = False) -> List[List[int]]:
        '''
        Return Pareto fronts as index lists,
        and update `indi.pareto_rank` in-place.
        '''
        n = db.size
        if n == 0:
            return []

        dominating_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        ys = None
        if is_valid_database:
            ys = EvolutionaryAlgorithm._get_unified_objectives(db)

        for p in range(n - 1):
            for q in range(p + 1, n):
                if is_valid_database:
                    flag = EvolutionaryAlgorithm.pareto_dominance(ys[p, :], ys[q, :])
                else:
                    flag = db.individuals[p].check_dominance(db.individuals[q])
                if flag == 1:
                    dominated_set[p].append(q)
                    dominating_count[q] += 1
                elif flag == -1:
                    dominated_set[q].append(p)
                    dominating_count[p] += 1

        for i in range(n):
            if dominating_count[i] == 0:
                db.individuals[i].pareto_rank = 1
                fronts[0].append(i)

        i_front = 0
        while i_front < len(fronts) and len(fronts[i_front]) > 0:
            next_front: List[int] = []
            for p in fronts[i_front]:
                for q in dominated_set[p]:
                    dominating_count[q] -= 1
                    if dominating_count[q] == 0:
                        db.individuals[q].pareto_rank = i_front + 2
                        next_front.append(q)
            if len(next_front) > 0:
                fronts.append(next_front)
            i_front += 1

        return fronts

    @staticmethod
    def assign_crowding_distance(db: Database, fronts: List[List[int]]) -> None:
        '''
        NSGA-II crowding distance on each front (Deb et al.).
        '''
        problem = db.problem
        for front in fronts:
            if len(front) == 0:
                continue

            for idx in front:
                db.individuals[idx].crowding_distance = 0.0

            if len(front) <= 2:
                for idx in front:
                    db.individuals[idx].crowding_distance = float("inf")
                continue

            n_obj = problem.n_objective
            obj_matrix = np.zeros((len(front), n_obj))
            for i_local, idx in enumerate(front):
                obj_matrix[i_local, :] = db.individuals[idx].objectives

            sign = []
            for out_type in problem.problem_settings.output_type:
                if abs(out_type) == 1:
                    sign.append(-1.0 if out_type == 1 else 1.0)
            sign = np.array(sign, dtype=float)
            obj_matrix = obj_matrix * sign

            for m in range(n_obj):
                order = np.argsort(obj_matrix[:, m])
                sorted_values = obj_matrix[order, m]
                f_min = sorted_values[0]
                f_max = sorted_values[-1]
                span = max(f_max - f_min, 1.0e-12)

                db.individuals[front[order[0]]].crowding_distance = float("inf")
                db.individuals[front[order[-1]]].crowding_distance = float("inf")

                for i in range(1, len(front) - 1):
                    idx_mid = front[order[i]]
                    if np.isinf(db.individuals[idx_mid].crowding_distance):
                        continue
                    db.individuals[idx_mid].crowding_distance += (
                        sorted_values[i + 1] - sorted_values[i - 1]
                    ) / span

    @staticmethod
    def select_population_indices(db: Database, fronts: List[List[int]],
                    population_size: int) -> List[int]:
        '''
        Environmental selection: fill by rank, then crowding distance on the last partial front.
        '''
        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
                continue

            remaining = population_size - len(selected)
            if remaining > 0:
                sorted_front = sorted(
                    front,
                    key=lambda idx: db.individuals[idx].crowding_distance,
                    reverse=True,
                )
                selected.extend(sorted_front[:remaining])
            break

        return selected

    #! Large computational cost (due to fast_non_dominated_ranking)
    @staticmethod
    def rank_pareto(db: Database, is_valid_database: bool = False) -> None:
        '''
        Assign Pareto rank + crowding distance, then sort database.
        
        Parameters:
        -----------
        db: Database
            Database to rank.
        is_valid_database: bool
            Whether `db` is a valid database.
        '''
        fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(
            db, is_valid_database=is_valid_database)
        EvolutionaryAlgorithm.assign_crowding_distance(db, fronts)
        db.sort_database(sort_type=0)

    #! Large computational cost (due to rank_pareto)
    @staticmethod
    def save_elite(db: Database, elite: Database,
                    is_db_valid: bool = False) -> None:
        '''
        Save rank1 individuals of db to the elite database (can be empty). 
        Then sort elite, and only keep the updated rank 1 individuals.
        
        Note that `db` and `elite` will be sorted.
        
        Parameters:
        -----------
        db: Database
            Database to save elite from.
        elite: Database
            Elite database to save individuals to.
        is_db_valid: bool
            Whether `db` is a valid database.
        '''
        if db.size <= 0:
            elite.individuals = []
            elite.update_id_list()
            return

        if not db.sorted:
            EvolutionaryAlgorithm.rank_pareto(db, is_valid_database=is_db_valid)

        rank1_indices: List[int] = []
        for i, indi in enumerate(db.individuals):
            if indi.pareto_rank == 1:
                rank1_indices.append(i)
            else:
                break

        elite.individuals = [db.individuals[i] for i in rank1_indices]
        elite.update_id_list()
        elite._sorted = False
        elite.sort_database(sort_type=0)

    @staticmethod
    def add_bests(db: Database, other: Database, num: int = 0) -> None:
        '''
        Add best individuals from `db` into `other`.
        
        Parameters:
        -----------
        db: Database
            Database to add best individuals from.
        other: Database
            Database to add best individuals to.
        num: int
            Number of best individuals to add.
        '''
        if db.size <= 0:
            return

        temp = db.get_sub_database(index_list=list(range(db.size)), deepcopy=True)
        EvolutionaryAlgorithm.rank_pareto(temp, is_valid_database=False)

        if num <= 0:
            count = 0
            for indi in temp.individuals:
                if indi.pareto_rank == 1:
                    count += 1
                else:
                    break
            num = count
        num = min(num, temp.size)

        for i in range(num):
            other.add_individual(
                temp.individuals[i],
                check_duplication=True,
                check_bounds=True,
                deepcopy=False, # individuals are already copied in temp
            )

    @staticmethod
    def push_into_population(individuals: List, pop: Database) -> None:
        '''
        Push individuals into `pop`,
        replacing the last individuals of `pop`.
        
        Parameters:
        -----------
        individuals: List
            List of individuals to push into `pop`.
        pop: Database
            Database to push individuals into.
        '''
        n_added = 0
        n0 = pop.size
        for indi in individuals:
            if pop.add_individual(indi, 
                    check_duplication=True, 
                    check_bounds=True,
                    deepcopy=True):
                n_added += 1

        remove_ids = []
        for i in range(n_added):
            remove_ids.append(pop.individuals[n0 - 1 - i].ID)

        for ID in remove_ids:
            pop.delete_individual(ID=ID)
            
        pop.update_id_list()

    @staticmethod
    def shrink_population(pop: Database, population_size: int, 
                reserve_ratio: float = 0.3) -> None:
        '''
        Shrink population to `population_size` by deleting
        worst individuals (based on Pareto rank and crowding distance).
        
        Parameters:
        -----------
        pop: Database
            Population to shrink.
        population_size: int
            Population size to shrink to.
        reserve_ratio: float
            Reserve ratio for population shrinking, i.e.,
            ratio of individuals that are directly kept.
        '''
        if pop.size < population_size:
            return

        if not pop.sorted:
            EvolutionaryAlgorithm.rank_pareto(pop)

        n_pop = pop.size
        n_direct = int(reserve_ratio * population_size)
        if n_direct > 0:
            ii_sub = np.random.choice(n_pop, size=n_direct, replace=False)
            id_direct = [pop.individuals[i].ID for i in ii_sub]
        else:
            id_direct = []

        i = 1
        while pop.size > population_size:
            ID0 = pop.individuals[-i].ID
            if ID0 in id_direct:
                i += 1
            else:
                pop.delete_individual(ID=ID0)


class OptEvolutionaryFramework(OptBaseFramework):
    '''
    Base optimization framework using evolutionary optimization algorithms.
    
    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    evolutionary_algorithm: EvolutionaryAlgorithm
        Evolutionary algorithm to use.
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
        Total database, including all individuals.
    db_valid: Database
        Valid database, including only valid individuals.
    db_elite: Database
        Elite database, including only elite individuals.
    db_candidate: Database
        Population database, including only candidate individuals.
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
            evolutionary_algorithm: EvolutionaryAlgorithm = None,
            user_func: Callable = None,
            mp_evaluation: MultiProcessEvaluation = None):
        
        super().__init__(
                problem=problem, optimization_settings=optimization_settings,
                user_func=user_func, mp_evaluation=mp_evaluation)

        if evolutionary_algorithm is None:
            self.evolutionary_algorithm = EvolutionaryAlgorithm()
        else:
            self.evolutionary_algorithm = evolutionary_algorithm

    def select_elite_from_valid(self) -> None:
        '''
        Select elite set from valid database.
        '''
        if self.db_valid.size <= 0:
            self.db_elite = Database(self.problem, database_type="elite")
            return

        fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(
            db=self.db_valid, is_valid_database=True)
        
        EvolutionaryAlgorithm.assign_crowding_distance(
            db=self.db_valid, fronts=fronts)

        n_take = min(self.population_size, self.db_valid.size)
        selected_indices = EvolutionaryAlgorithm.select_population_indices(
            db=self.db_valid, fronts=fronts, population_size=n_take)
        
        sub_valid = self.db_valid.get_sub_database(
            index_list=selected_indices, deepcopy=True)
        sub_valid.sort_database(sort_type=0)

        if len(fronts) > 0 and len(fronts[0]) > 0:
            self.db_elite = sub_valid.get_sub_database(
                index_list=fronts[0], deepcopy=True)
            self.db_elite.sort_database(sort_type=0)
        else:
            self.db_elite = Database(self.problem, database_type="elite")
