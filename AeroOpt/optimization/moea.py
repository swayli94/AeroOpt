'''
Class for multi-objective evolutionary algorithms.

- Dominance-based (Pareto-based) algorithms, e.g., NSGA-II, NSGA-III, NSDE, SPEA2, GDE3, etc.
- Decomposition-based algorithms, e.g., MOEA/D, RVEA, etc.
- Indicator-based algorithms, e.g., SMS-EMOA, IBEA, etc.
- Approximation-guided algorithms, e.g., AGE-MOEA, etc.
'''

import math
import copy
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
from AeroOpt.core import Database


class Algorithm(ABC):
    '''
    Base class for optimization algorithms.
    
    This class provides:
    - Selection of elite individuals from the valid database (optional).
    - Building a temporary parent database (optional).
    - Generation of candidate individuals as child generation (mandatory).

    '''
    @staticmethod
    def build_temporary_parent_database(
            db_valid: Database,
            population_size: int,
            **kwargs: Any,
            ) -> Database:
        '''
        Optional hook for building a temporary parent pool from the valid database.
        The `da_valid` is deep copied.
        
        Parameters:
        -----------
        db_valid: Database
            Valid database.
        population_size: int
            Size of the parent pool.
        **kwargs: Any
            Additional keyword arguments.
            
        Returns:
        --------
        db_parent: Database
            Temporary parent pool.
        '''
        ...
    
    @staticmethod
    def environmental_selection_indices(
            db: Database,
            population_size: int,
            **kwargs: Any,
            ) -> List[int]:
        '''
        Optional hook for environmental selection on a merged pool.
        Get indices of individuals to keep (length <= `population_size`).
        
        Parameters:
        -----------
        db: Database
            Database to select from.
        population_size: int
            Size of the parent pool.
        **kwargs: Any
            Additional keyword arguments.
            
        Returns:
        --------
        index_selected: List[int]
            Indices of selected individuals, length at most `population_size`.
        '''
        ...

    @staticmethod
    @abstractmethod
    def generate_candidate_individuals(
            db_valid: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            **kwargs: Any,
            ) -> None:
        '''
        Generate candidate individuals during the optimization,
        which are stored in `db_candidate` database before evaluation.
        The `db_candidate` database is generated from `db_valid` database:
        
        - update `db_candidate` in place.
        - create a temporary parent database by selection from `db_valid`
        - evolution (crossover, mutation, etc.) of the parent database
        
        Parameters:
        -----------
        db_valid: Database
            Valid database.
        db_candidate: Database
            Candidate database.
        population_size: int
            Size of the parent pool.
        iteration: int
            Current iteration.
        **kwargs: Any
            Additional keyword arguments.
            
        Returns:
        --------
        None
        '''
        ...


class DominanceBasedAlgorithm(object):
    '''
    Dominance-based (Pareto-based) multi-objective evolutionary algorithms,
    e.g., NSGA-II, NSGA-III, NSDE, SPEA2, GDE3, etc.

    This class provides:
    - Pareto non-dominated ranking.
    - Crowding-distance assignment.
    - Selection of parent pool.
    - Selection of elite individuals from the valid database.
    '''
    @staticmethod
    def check_pareto_dominance(y: np.ndarray, y_other: np.ndarray) -> int:
        '''
        Pareto dominance on already direction-unified objective vectors.

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
        dominance_list = np.sign(y - y_other).tolist()

        if 1 in dominance_list and -1 in dominance_list:
            return 9
        if 1 in dominance_list and -1 not in dominance_list:
            return 1
        if 1 not in dominance_list and -1 in dominance_list:
            return -1
        return 0

    @staticmethod
    def non_dominated_ranking(db: Database) -> List[List[int]]:
        '''
        Non-dominated ranking on the database:
        
        - suggested to apply to a valid database, i.e., `db.is_valid_database` is True.
        - update `indi.pareto_rank` in-place.
        
        Parameters:
        -----------
        db: Database
            Database to rank.
            
        Returns:
        --------
        index_fronts: List[List[int]]
            Pareto fronts as index lists.
        '''
        n = db.size
        if n == 0:
            return []
        
        dominating_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        index_fronts: List[List[int]] = [[]]

        if db.is_valid_database:
            scaled_ys = db.get_unified_objectives(scale=True)
        else:
            scaled_ys = None

        for p in range(n - 1):
            for q in range(p + 1, n):
                if scaled_ys is not None:
                    flag = DominanceBasedAlgorithm.check_pareto_dominance(scaled_ys[p, :], scaled_ys[q, :])
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
                index_fronts[0].append(i)

        i_front = 0
        while i_front < len(index_fronts) and len(index_fronts[i_front]) > 0:
            next_front: List[int] = []
            for p in index_fronts[i_front]:
                for q in dominated_set[p]:
                    dominating_count[q] -= 1
                    if dominating_count[q] == 0:
                        db.individuals[q].pareto_rank = i_front + 2
                        next_front.append(q)
            if len(next_front) > 0:
                index_fronts.append(next_front)
            i_front += 1

        db._updated_pareto_rank = True
        db._index_pareto_fronts = copy.deepcopy(index_fronts)

        return index_fronts

    @staticmethod
    def assign_crowding_distance(db: Database) -> None:
        '''
        Assign crowding distance on each front (Deb et al.).
        '''
        problem = db.problem
        n_objective = problem.n_objective
        scaled_ys = db.get_unified_objectives(scale=True) # [nn, n_objective]
        
        # Empty crowding distance
        for indi in db.individuals:
            indi.crowding_distance = 0.0
        
        for front in db._index_pareto_fronts:
            
            if len(front) == 0:
                continue

            if len(front) <= 2:
                for idx in front:
                    db.individuals[idx].crowding_distance = float("inf")
                continue

            obj_matrix = scaled_ys[front, :] # [nf, n_objective]

            for m in range(n_objective):
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

        db._updated_crowding_distance = True

    @staticmethod
    def select_parent_indices(db: Database, n_select: int) -> List[int]:
        '''
        Selection of parent pool (truncation without recomputation).

        Iterates over Pareto fronts in non-dominated order (best front first).
        Each full front is kept entirely while the running count stays at or
        below `n_select`. When the next front would exceed `n_select`,
        only `remaining` individuals are taken from that front: those with
        the largest `crowding_distance` values (ties broken by index order
        from `sorted`), which favours sparser regions of the objective space.

        Deeper fronts after a partial acceptance are ignored.

        Parameters:
        -----------
        db : Database
            Population whose `individuals[idx].crowding_distance` must already
            be assigned (e.g. by `assign_crowding_distance`).
        n_select : int
            Target number of selected individual indices.

        Returns:
        --------
        index_selected: List[int]
            Indices of selected individuals, length at most `n_select`.
        '''
        index_selected: List[int] = []
        for front in db._index_pareto_fronts:
            if len(index_selected) + len(front) <= n_select:
                index_selected.extend(front)
                continue

            remaining = n_select - len(index_selected)
            if remaining > 0:
                sorted_front = sorted(
                    front,
                    key=lambda idx: db.individuals[idx].crowding_distance,
                    reverse=True,
                )
                index_selected.extend(sorted_front[:remaining])
            break

        return index_selected

    @staticmethod
    def rank_pareto(db: Database) -> None:
        '''
        Assign Pareto rank and crowding distance, then sort database.

        After reordering individuals, remap `db._index_pareto_fronts` so indices
        still refer to the same individuals (by ID); otherwise any later use of
        `select_parent_indices` would read stale positions.
        '''
        if db.size <= 0:
            return
        DominanceBasedAlgorithm.non_dominated_ranking(db)
        DominanceBasedAlgorithm.assign_crowding_distance(db)
        ids_before = [indi.ID for indi in db.individuals]
        index_fronts_before = copy.deepcopy(db._index_pareto_fronts)
        db.sort_database(sort_type=0)
        id_to_new_idx = {indi.ID: i for i, indi in enumerate(db.individuals)}
        db._index_pareto_fronts = [
            [id_to_new_idx[ids_before[i]] for i in front]
            for front in index_fronts_before
        ]

    @staticmethod
    def build_temporary_parent_database(db_valid: Database, population_size: int) -> Database:
        '''
        Build a temporary parent pool from the valid database:
        
        - Non-dominated ranking
        - Crowding distance assignment
        - Selection of parent pool
        
        The db_valid is updated in-place.
        
        Parameters:
        -----------
        db_valid: Database
            Valid database.
        population_size: int
            Size of the parent pool.
            
        Returns:
        --------
        db_parent: Database
            Temporary parent pool.
        '''
        if not db_valid.is_valid_database:
            raise ValueError("Needs a valid database (db_valid.is_valid_database=True).")
        
        if db_valid.size <= 0:
            raise ValueError("Cannot build parent database from an empty valid database.")

        if not db_valid.updated_pareto_rank:
            DominanceBasedAlgorithm.non_dominated_ranking(db_valid)
        
        DominanceBasedAlgorithm.assign_crowding_distance(db_valid)
        
        if db_valid.size <= population_size:
            return db_valid
        
        index_list = DominanceBasedAlgorithm.select_parent_indices(
            db_valid, population_size)
        
        db_parent = db_valid.get_sub_database(index_list=index_list, deepcopy=True)
        
        return db_parent
    
    @staticmethod
    def select_elite_from_valid(db_valid: Database, db_elite: Database) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        if db_valid.size <= 0:
            db_elite.empty_database()
            return

        DominanceBasedAlgorithm.non_dominated_ranking(db_valid)
        
        if len(db_valid._index_pareto_fronts) <= 0 or len(db_valid._index_pareto_fronts[0]) <= 0:
            db_elite.empty_database()
            return
        
        DominanceBasedAlgorithm.assign_crowding_distance(db_valid)

        db_elite.copy_from_database(db_valid, 
                                index_list=db_valid._index_pareto_fronts[0], 
                                deepcopy=True)
        db_elite.sort_database(sort_type=1)
    

class DecompositionBasedAlgorithm(object):
    '''
    Decomposition-based multi-objective evolutionary algorithms,
    e.g., MOEA/D, RVEA, etc.
    '''
    @staticmethod
    def suggest_n_partitions(n_objective: int, population_size: int) -> int:
        '''
        Pick a simplex partition count so the number of reference points is near
        `population_size` (combinatorial count C(p+M-1, M-1)).
        '''
        if n_objective <= 1:
            return 1
        best_p, best_d = 1, float('inf')
        pop = max(1, int(population_size))
        # Upper search bound: bi-objective needs p ≈ pop-1; many-objective uses smaller p.
        hi = max(41, pop + 25)
        for p in range(1, hi):
            n_ref = math.comb(p + n_objective - 1, n_objective - 1)
            d = abs(n_ref - pop)
            if d < best_d:
                best_d, best_p = d, p
        return max(1, best_p)

    @staticmethod
    def das_dennis_reference_points(n_objective: int, n_partitions: int) -> np.ndarray:
        '''
        Das-Dennis reference points on the (n_objective-1)-simplex, shape [n_ref, n_objective].
        '''
        if n_objective <= 0:
            return np.zeros((0, 0))
        if n_objective == 1:
            return np.ones((1, 1))
        n_partitions = max(1, int(n_partitions))
        out: List[List[int]] = []

        def rec(left: int, m: int, prefix: List[int]) -> None:
            if m == 0:
                out.append(prefix + [left])
                return
            for i in range(left + 1):
                rec(left - i, m - 1, prefix + [i])

        rec(n_partitions, n_objective - 1, [])
        return np.asarray(out, dtype=float) / float(n_partitions)
