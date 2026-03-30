'''
Class for multi-objective evolutionary algorithms.

- Dominance-based (Pareto-based) algorithms, e.g., NSGA-II, NSGA-III, RVEA, etc.
- Decomposition-based algorithms, e.g., MOEA/D, etc.
- Indicator-based algorithms, e.g., SMS-EMOA, IBEA, etc.
- Approximation-guided algorithms, e.g., AGE-MOEA, etc.
'''

import math
import copy
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import numpy as np
from aeroopt.core import Database


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
    e.g., NSGA-II, NSGA-III, RVEA, NSDE, SPEA2, GDE3, etc.

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
        
        for front in db.index_pareto_fronts:
            
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
        for front in db.index_pareto_fronts:
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
        index_fronts_before = copy.deepcopy(db.index_pareto_fronts)
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
        
        if len(db_valid.index_pareto_fronts) <= 0 or len(db_valid.index_pareto_fronts[0]) <= 0:
            db_elite.empty_database()
            return
        
        DominanceBasedAlgorithm.assign_crowding_distance(db_valid)

        db_elite.copy_from_database(db_valid, 
                                index_list=db_valid.index_pareto_fronts[0], 
                                deepcopy=True)
        db_elite.sort_database(sort_type=1)
    

class DecompositionBasedAlgorithm(object):
    '''
    Decomposition-based multi-objective evolutionary algorithms,
    e.g., MOEA/D, etc.
    
    Note that other algorithms, such as NSGA-III, RVEA,
    also use this class for reference points generation.
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
            n_partitions = math.comb(p + n_objective - 1, n_objective - 1)
            d = abs(n_partitions - pop)
            if d < best_d:
                best_d, best_p = d, p
        return max(1, best_p)

    @staticmethod
    def das_dennis_reference_points(n_objective: int, n_partitions: int) -> np.ndarray:
        '''
        Das-Dennis reference points on the (n_objective-1)-simplex.
        
        Parameters:
        -----------
        n_objective: int
            Number of objectives.
        n_partitions: int
            Number of reference points (subproblems).
            
        Returns:
        --------
        ref_points: np.ndarray [n_partitions, n_objective]
            Das-Dennis reference points (weight vectors) on the (n_objective-1)-simplex.
            Each row is a weight vector, sum of the row is 1.
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

    @staticmethod
    def default_decomposition_name(n_objective: int) -> str:
        '''
        Default decomposition method:
        - Tchebycheff for <=2 objectives,
        - PBI for >2 objectives.
        '''
        return 'tchebicheff' if int(n_objective) <= 2 else 'pbi'

    @staticmethod
    def decomposed_values(ys: np.ndarray, weights: np.ndarray,
            ideal: np.ndarray, method: str, pbi_theta: float = 5.0) -> np.ndarray:
        '''
        Compute scalarized objective values for MOEA/D subproblems.

        In MOEA/D, each subproblem is defined by a reference direction (weight vector).
        Return values form a matrix of size `(n_points, n_weights)`: entry `(i, j)`
        is the scalarization of point `i` under weight `j` (smaller is better).

        Two decomposition methods are supported:

        1. Tchebycheff:
            
            Emphasizes the worst (most deviated) objective.
            This promotes balanced improvement across objectives and is commonly used
            for low-dimensional objective spaces.

        2. Penalty-based Boundary Intersection (PBI):
        
            Decomposes the objective vector into:
            d1: distance along the reference direction (convergence).
            d2: perpendicular distance to the direction (diversity).

            The scalar value is a combination of the convergence and diversity,
            where `pbi_theta` controls the trade-off between convergence and diversity.
            This method is more suitable for higher-dimensional objective spaces.

        Parameters
        ----------
        ys : np.ndarray [n_points, n_objective]
            Unified scaled objective values of candidate solutions.
            A single point may be given as shape `(n_objective,)`.
        weights : np.ndarray [n_weights, n_objective]
            Reference directions (weight vectors).
            A single weight may be given as shape `(n_objective,)`.
        ideal : np.ndarray [n_objective]
            Scaled ideal point (best observed value for each objective).
        method : str
            Decomposition method to use: 'tchebicheff' or 'pbi'.
        pbi_theta : float
            Penalty parameter for PBI, controlling the balance between
            convergence (d1) and diversity (d2).

        Returns
        -------
        scalars : np.ndarray [n_points, n_weights]
            `scalars[i, j]` is the value for `ys[i]` under `weights[j]`.
            When `n_points == n_weights` and row `i` uses weight `i` (neighbor
            alignment in MOEA/D), take `np.diag(scalars)`.

        Notes
        -----
        One objective row or one weight row can be passed as 1D; they are promoted
        with `numpy.atleast_2d` to shapes `(1, n_objective)` / `(1, n_objective)`.
        '''
        ys = np.atleast_2d(np.asarray(ys, dtype=float))
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        ideal = np.asarray(ideal, dtype=float).reshape(-1)

        if ys.ndim != 2 or weights.ndim != 2:
            raise ValueError('ys and weights must be 1D or 2D arrays.')
        _, m_y = ys.shape
        _, m_w = weights.shape
        if m_y != m_w:
            raise ValueError(
                f'Objective dimension mismatch: ys has {m_y}, weights has {m_w}.')
        if ideal.shape[0] != m_y:
            raise ValueError(
                f'ideal length {ideal.shape[0]} != n_objective {m_y}.')

        lam_n = np.maximum(weights, 1.0e-32)
        row_norm = np.linalg.norm(lam_n, axis=1, keepdims=True)
        row_norm = np.maximum(row_norm, 1.0e-32)
        lam_unit = lam_n / row_norm

        diff = ys - ideal[None, :]
        diff_exp = diff[:, None, :]
        lam_n_exp = lam_n[None, :, :]

        if method == 'tchebicheff':
            return np.max(lam_n_exp * np.abs(diff_exp), axis=2)

        if method == 'pbi':
            lam_u_exp = lam_unit[None, :, :]
            d1 = np.sum(diff_exp * lam_u_exp, axis=2)
            d1 = np.maximum(d1, 0.0)
            norm_l = np.linalg.norm(lam_unit, axis=1)
            norm_l = np.maximum(norm_l, 1.0e-32)
            proj = (d1 / norm_l[None, :])[:, :, None] * lam_unit[None, :, :]
            d2 = np.linalg.norm(diff_exp - proj, axis=2)
            return d1 + float(pbi_theta) * d2

        raise ValueError(f'Unknown decomposition method: {method}')

    @staticmethod
    def _pareto_first_front_mask(ys: np.ndarray) -> np.ndarray:
        '''
        First non-dominated front for minimization, same point set as
        `DominanceBasedAlgorithm.non_dominated_ranking` front 0.
        '''
        ys = np.asarray(ys, dtype=float)
        n = ys.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            le = np.all(ys <= ys[i], axis=1)
            st = np.any(ys < ys[i], axis=1)
            dominates_i = le & st
            dominates_i[i] = False
            if np.any(dominates_i):
                dominated[i] = True
        return ~dominated

    @staticmethod
    def reference_direction_progress(
            ys: np.ndarray,
            n_partitions: int,
            pareto_front_only: bool = True,
            decomposition: str = 'auto',
            pbi_theta: float = 5.0,
            ideal: Optional[np.ndarray] = None,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        For each Das-Dennis reference direction `lambda_j`, compute the best
        (minimum) MOEA/D-style scalarization :math:`g(f\\mid\\lambda_j, z^*)`
        achieved on the analyzed objective rows. **Larger** best values mean
        no analyzed point approaches :math:`z^*` well along that preference
        direction --- a comparatively **slow** or **lagging** direction on the
        current (approximate) Pareto front.

        This is intended for **post-hoc analysis** of any non-dominated set,
        not for algorithm-internal selection.

        Parameters
        ----------
        ys : np.ndarray [n_points, n_objective]
            Unified scaled objective matrix for minimization.
        n_partitions : int
            Simplex partition count for `das_dennis_reference_points`.
        pareto_front_only : bool
            If True, evaluate the progress on the first non-dominated front of `ys`.
        decomposition : {'auto', 'tchebicheff', 'pbi'}
            Scalarization; `'auto'` uses Tchebycheff for `n_objective <= 2`
            else PBI (same default spirit as MOEA/D in this project).
        pbi_theta : float
            PBI penalty parameter when using `decomposition='pbi'`.
        ideal : np.ndarray [n_objective]
            Scaled ideal point `z_star` for scalarization.
            If omitted, use the component-wise minimum of the analyzed rows.

        Returns
        -------
        ordered_ref_indices : np.ndarray [n_partitions]
            Indices `j` into rows of `reference_points`, **slowest first**
            (descending `best_achievement`).
        best_achievement : np.ndarray [n_partitions]
            Per-direction best scalar :math:`\\min_i g(f_i\\mid\\lambda_j)`,
            aligned with `reference_points`.
        reference_points : np.ndarray [n_partitions, n_objective]
            Das-Dennis reference points.
        '''
        ys = np.asarray(ys, dtype=float)
        if ys.ndim != 2 or ys.shape[0] == 0:
            return (
                np.array([], dtype=int),
                np.array([]),
                np.zeros((0, 0), dtype=float),
            )
        n_obj = ys.shape[1]
        if n_obj < 2:
            ref = DecompositionBasedAlgorithm.das_dennis_reference_points(1, n_partitions)
            return (
                np.array([0], dtype=int),
                np.zeros(1, dtype=float),
                ref,
            )

        if pareto_front_only:
            mask = DecompositionBasedAlgorithm._pareto_first_front_mask(ys)
            ys_work = ys[mask]
        else:
            ys_work = ys

        if ys_work.shape[0] == 0:
            ref = DecompositionBasedAlgorithm.das_dennis_reference_points(n_obj, n_partitions)
            return (
                np.arange(ref.shape[0], dtype=int),
                np.full(ref.shape[0], np.nan),
                ref,
            )

        if ideal is None:
            z_star = ys_work.min(axis=0).astype(float)
        else:
            z_star = np.asarray(ideal, dtype=float).reshape(-1)
            if z_star.shape[0] != n_obj:
                raise ValueError(
                    f'ideal length {z_star.shape[0]} != n_objective {n_obj}')

        dec = decomposition
        if dec == 'auto':
            dec = DecompositionBasedAlgorithm.default_decomposition_name(n_obj)
        if dec not in ('tchebicheff', 'pbi'):
            raise ValueError(
                f"decomposition must be 'auto', 'tchebicheff', or 'pbi', got {dec!r}")

        ref = DecompositionBasedAlgorithm.das_dennis_reference_points(
            n_obj, n_partitions)
        g_mat = DecompositionBasedAlgorithm.decomposed_values(
            ys_work, ref, z_star, dec, float(pbi_theta)) # [n_points, n_partitions]
        best_g = np.min(g_mat, axis=0) # [n_partitions]

        ordered = np.argsort(-best_g, kind='stable')
        return ordered, best_g, ref

    @staticmethod
    def find_slow_directions(
            db: Database,
            n_partitions: int,
            pareto_front_only: bool = True,
            decomposition: str = 'auto',
            pbi_theta: float = 5.0,
            ideal: Optional[np.ndarray] = None,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Same analysis as `reference_direction_progress`, using objectives
        from `db`.

        Parameters
        ----------
        db : Database
            Database with valid unified scaled objectives (`get_unified_objectives`).
        n_partitions : int
            Number of reference points (subproblems).
        pareto_front_only : bool
            If True, evaluate the progress on the first non-dominated front of `ys`.
        decomposition, pbi_theta, ideal
            Passed through to `reference_direction_progress`.

        Returns
        -------
        ordered_ref_indices : np.ndarray [n_partitions]
            Slowest directions first (same as `reference_direction_progress`).
        best_achievement : np.ndarray [n_partitions]
            Per-direction best scalarized value on the analyzed set.
        reference_points : np.ndarray [n_partitions, n_objective]
            Das-Dennis reference points (weight vectors) on the (n_objective-1)-simplex.
            Each row is a weight vector, sum of the row is 1.
        '''
        ys = db.get_unified_objectives(scale=True)
        if ys.size == 0 or ys.shape[1] < 1:
            return (
                np.array([], dtype=int),
                np.array([]),
                np.zeros((0, max(1, db.problem.n_objective)), dtype=float),
            )
        return DecompositionBasedAlgorithm.reference_direction_progress(
            ys,
            n_partitions,
            pareto_front_only=pareto_front_only,
            decomposition=decomposition,
            pbi_theta=pbi_theta,
            ideal=ideal,
        )

