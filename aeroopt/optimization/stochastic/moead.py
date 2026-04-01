'''
MOEA/D: multiobjective evolutionary algorithm based on decomposition.

Each subproblem is associated with a weight vector on the objective simplex;
offspring are generated from parents chosen in the neighborhood of weights (or
globally with small probability). Scalar fitness is given by a decomposition
method (Tchebycheff or PBI), matching the spirit of pymoo's MOEA/D.

References:

    Qingfu Zhang and Hui Li. A multi-objective evolutionary algorithm based on decomposition.
    IEEE Transactions on Evolutionary Computation, Accepted, 2007.

    https://pymoo.org/algorithms/moo/moead.html

    https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/moead.py
'''

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from aeroopt.core import (
    Problem,
    Individual,
    Database,
    MultiProcessEvaluation,
)

from aeroopt.optimization.moea import (
    Algorithm,
    DominanceBasedAlgorithm,
    DecompositionBasedAlgorithm,
)
from aeroopt.optimization.utils import (
    polynomial_mutation,
    sbx_crossover,
)
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.settings import (
    SettingsMOEAD,
    SettingsOptimization,
)


class MOEAD(Algorithm):
    '''
    MOEA/D operators: weights, neighborhoods, decomposition, mating and
    subproblem replacement.
    '''
    @staticmethod
    def _select_parent_slots(
            k: int,
            neighbors: np.ndarray,
            n_pop: int,
            n_parents: int,
            prob_neighbor: float,
            rng: np.random.Generator,
            ) -> np.ndarray:
        '''
        Choose parent slot indices for subproblem `k` used in crossover.
        
        Parameters:
        -----------
        k: int
            Index of the subproblem.
        neighbors: np.ndarray
            Neighbors of the subproblem.
        n_pop: int
            Total number of subproblems.
        n_parents: int
            Number of parents to select.
        prob_neighbor: float
            Probability of drawing parents from the neighborhood.
        rng: np.random.Generator
            Random number generator.
            
        Returns:
        --------
        parent_slots: np.ndarray
            Indices into the current population of subproblems.
        '''
        if rng.random() < prob_neighbor:
            pool = neighbors[k, :]
            return rng.choice(pool, size=n_parents, replace=False)
        return rng.choice(n_pop, size=n_parents, replace=False)

    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            ref_dirs: np.ndarray = np.array([]),
            neighbors: np.ndarray = np.array([]),
            slot_ids: np.ndarray = np.array([]),
            prob_neighbor: float = 0.0,
            decomposition_method: str = 'auto',
            pbi_theta: float = 5.0,
            ideal: np.ndarray = np.array([]),
            cross_rate: float = 0.8,
            pow_sbx: float = 20.0,
            mut_rate: float = 1.0,
            pow_poly: float = 20.0,
            rng: np.random.Generator = np.random.default_rng(),
            pending_list: List[Tuple[int, int]] = [],
            ) -> None:
        '''
        Generate one offspring per subproblem in random order (MOEA/D parallel
        offspring scheme): SBX crossover between parents from neighborhood or
        global pool, then polynomial mutation. Clears `db_candidate` and appends
        `(subproblem_index, offspring_ID)` to `pending_list` for neighbor
        replacement after evaluation. `decomposition_method`, `pbi_theta`, and
        `ideal` are accepted for API symmetry; this routine does not use them.
        
        Parameters:
        -----------
        db: Database
            Population database.
        db_candidate: Database
            Candidate database.
        population_size: int
            Size of the parent pool (not used in MOEA/D).
        iteration: int
            Current iteration.
        ref_dirs: np.ndarray
            Reference directions.
        neighbors: np.ndarray
            Neighbors of the subproblem.
        slot_ids: np.ndarray
            Indices into the current population of subproblems.
        prob_neighbor: float
            Probability of drawing parents from the neighborhood.
        decomposition_method: str
            Decomposition method.
        pbi_theta: float
            PBI theta parameter.
        ideal: np.ndarray
            Ideal point.
        cross_rate: float
            Crossover rate.
        pow_sbx: float
            SBX power.
        mut_rate: float
            Mutation rate.
        pow_poly: float
            Polynomial power.
        rng: np.random.Generator
            Random number generator.
        pending_list: List[Tuple[int, int]]
            List of (subproblem_index, offspring_ID) to be replaced.
        '''
        _ = decomposition_method, pbi_theta, ideal

        n_pop = int(ref_dirs.shape[0])
        if db.size <= 0:
            raise RuntimeError("No individuals available for MOEA/D.")

        db_candidate.empty_database()
        pending_list.clear()

        order = rng.permutation(n_pop)
        n_parents = 2

        for k in order:
            p_slots = MOEAD._select_parent_slots(
                int(k), neighbors, n_pop, n_parents,
                prob_neighbor, rng)
            id1 = int(slot_ids[p_slots[0]])
            id2 = int(slot_ids[p_slots[1]])
            i1 = db.get_index_from_ID(id1)
            i2 = db.get_index_from_ID(id2)
            x1, x2 = sbx_crossover(
                db.individuals[i1].x,
                db.individuals[i2].x,
                problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx, rng=rng)
            pick = x1 if rng.random() < 0.5 else x2
            pick = polynomial_mutation(
                pick, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly, rng=rng)

            indi = Individual(problem=db_candidate.problem, x=pick)
            indi.source = 'evolutionary_operator'
            indi.generation = int(iteration)
            added, _ = db_candidate.add_individual(
                indi, check_duplication=True, check_bounds=True,
                deepcopy=False, print_warning_info=False)
            if not added:
                continue
            pending_list.append((int(k), int(indi.ID)))

    @staticmethod
    def neighbor_indices(ref_dirs: np.ndarray, n_neighbors: int) -> np.ndarray:
        '''
        Compute the neighborhood structure of reference directions in MOEA/D.

        In MOEA/D, each reference direction defines a subproblem. Instead of
        treating all subproblems independently, a neighborhood is constructed
        so that information (e.g., mating and solution updates) is shared only
        among similar subproblems. Similarity is defined in the weight space.

        This function computes pairwise Euclidean distances between reference
        directions and, for each direction, selects the indices of its
        n_neighbors closest directions (including itself). The resulting
        neighborhood is used to:
        - restrict parent selection to nearby subproblems,
        - propagate solutions locally during replacement,
        - improve convergence efficiency while preserving diversity.

        Parameters
        ----------
        ref_dirs : np.ndarray [n_subproblems, n_objectives]
            Array of reference directions (weight vectors).
        n_neighbors : int
            Number of nearest neighbors to select for each reference direction.

        Returns
        -------
        neighbors: np.ndarray [n_subproblems, t]
            Indices of nearest neighbors for each reference direction.
            Each row contains the indices of the closest reference 
            directions sorted by distance.
            t = min(n_neighbors, n_subproblems).
        '''
        w = np.asarray(ref_dirs, dtype=float)
        n_subproblems = w.shape[0]
        t = max(1, min(int(n_neighbors), n_subproblems))
        diff = w[:, None, :] - w[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return np.argsort(dist, axis=1, kind='stable')[:, :t]

    @staticmethod
    def update_ideal(ideal: np.ndarray, ys_row: np.ndarray) -> np.ndarray:
        '''
        Update ideal point using one (scaled) objective row.
        
        Parameters:
        -----------
        ideal: np.ndarray [n_objective]
            Ideal point (scaled values).
        ys_row: np.ndarray [n_objective]
            (Scaled) objective row.
            
        Returns:
        --------
        ideal: np.ndarray [n_objective]
            Updated ideal point (scaled values).
        '''
        ys_row = np.asarray(ys_row, dtype=float)
        np.minimum(ideal, ys_row, out=ideal)
        return ideal


class OptMOEAD(OptBaseFramework):
    '''
    Driver wiring MOEA/D to `OptBaseFramework`: subproblem slots, ideal point,
    and pending replacement queue.
    '''
    def __init__(self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsMOEAD,
            user_func=None,
            user_func_supports_parallel: bool = False,
            mp_evaluation: MultiProcessEvaluation|None = None,
            save_result_files: bool = True,
            logging: bool = True,
            ):

        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            user_func=user_func,
            user_func_supports_parallel=user_func_supports_parallel,
            mp_evaluation=mp_evaluation,
            save_result_files=save_result_files,
            logging=logging,
        )

        self.algorithm_settings = algorithm_settings
        n_obj = int(self.problem.n_objective)
        if n_obj < 2:
            raise ValueError("MOEA/D requires at least two objectives.")

        p = self.algorithm_settings.n_partitions
        if p is None:
            p = DecompositionBasedAlgorithm.suggest_n_partitions(
                n_obj, self.population_size)
        ref = DecompositionBasedAlgorithm.das_dennis_reference_points(n_obj, p)
        if ref.shape[0] != self.population_size:
            raise ValueError(
                "MOEA/D: population_size must equal the number of Das-Dennis "
                f"reference points ({ref.shape[0]}) for n_partitions={p}. "
                "Adjust population_size or n_partitions in settings.")

        self._ref_dirs = np.asarray(ref, dtype=float)
        t = max(1, min(self.algorithm_settings.n_neighbors, self._ref_dirs.shape[0]))
        self._neighbors = MOEAD.neighbor_indices(self._ref_dirs, t)

        dec = self.algorithm_settings.decomposition
        if dec == 'auto':
            dec = DecompositionBasedAlgorithm.default_decomposition_name(n_obj)
        if dec not in ('tchebicheff', 'pbi'):
            raise ValueError(
                f"decomposition must be 'auto', 'tchebicheff', or 'pbi', got {dec!r}")
        self._decomposition = dec

        self._ideal: Optional[np.ndarray] = None
        self._slot_ids: Optional[np.ndarray] = None
        self._pending: List[Tuple[int, int]] = []
        self._rng = np.random.default_rng()

    def main(self) -> None:
        '''
        Run the base main loop, then apply pending neighbor replacements for the
        last generation so state matches per-offspring updates at a generation
        boundary (as in pymoo). May update `_slot_ids` and `_ideal` and clear
        `_pending`.
        '''
        super().main()
        if self._pending:
            self._ensure_state()
            assert self._ideal is not None and self._slot_ids is not None
            self._apply_pending_replacements()
            self._pending.clear()


    def _ensure_state(self) -> None:
        '''
        Before the first evolutionary iteration, bind each of `N` subproblem
        slots to a valid representative individual ID and set `_ideal` from
        their objectives. If `n_valid < N`, cycle through valid individuals
        with `i % n_valid` so every weight has a slot (some may share the same
        decision vector initially; neighborhood replacement diversifies later).
        Raises if there are no valid individuals.
        '''
        if self._slot_ids is not None:
            return
        n = self._ref_dirs.shape[0]
        if self.db_valid.size <= 0:
            raise ValueError(
                "MOEA/D needs at least one valid individual to initialize subproblem "
                f"slots; got {self.db_valid.size}.")
        n_v = int(self.db_valid.size)
        self._slot_ids = np.array(
            [self.db_valid.get_ID_from_index(i % n_v) for i in range(n)],
            dtype=np.int64,
        )
        self._ideal = self.db_valid.get_unified_objectives(
            scale=True, ID_list=self._slot_ids.tolist()).min(axis=0).astype(float)

    def _replace_subproblem(self, k: int, offspring_id: int) -> None:
        '''
        Replace slots among neighbors of subproblem `k` where the offspring 
        improves the decomposition scalar value.
        
        `self._slot_ids[j]` holds the representative individual ID for subproblem `j`.
        If the offspring is not in `db_valid` with a valid evaluation,
        `self._slot_ids` remains unchanged.
        
        Parameters:
        -----------
        k: int
            Index of the subproblem.
        offspring_id: int
            ID of the offspring.            
        '''
        oidx = self.db_valid.get_index_from_ID(int(offspring_id))
        off = self.db_valid.individuals[oidx]
        if not off.valid_evaluation:
            return
        
        if self._slot_ids is None:
            raise ValueError("Subproblem slots are not initialized.")
        if self._ideal is None:
            raise ValueError("Ideal point is not initialized.")

        Nloc = self._neighbors[k, :]
        ids = self._slot_ids[Nloc].tolist()
        F_nei = self.db_valid.get_unified_objectives(scale=True, ID_list=ids)
        F_off = self.db_valid.get_unified_objectives(scale=True, index_list=[oidx])
        w_nei = self._ref_dirs[Nloc, :]

        g_nei = DecompositionBasedAlgorithm.decomposed_values(
            F_nei, w_nei, self._ideal, self._decomposition, self.algorithm_settings.pbi_theta)
        fv_nei = np.asarray(np.diag(g_nei), dtype=float)
        g_off = DecompositionBasedAlgorithm.decomposed_values(
            F_off, w_nei, self._ideal, self._decomposition, self.algorithm_settings.pbi_theta)
        fv_off = np.asarray(g_off[0, :], dtype=float)

        better = np.where(fv_off < fv_nei)[0]
        for j in better:
            self._slot_ids[Nloc[int(j)]] = int(offspring_id)

    def _apply_pending_replacements(self) -> None:
        '''
        For each `(k, offspring_id)` in `pending` (same order as generation),
        update the ideal point from the offspring objectives and run neighbor
        replacement for subproblem `k`. Skip entries whose ID is missing from
        `db` or lacks a valid evaluation (e.g. infeasible offspring).
        '''
        if self._ideal is None:
            raise ValueError("Ideal point is not initialized.")
        
        for k, oid in self._pending:
            oid_i = int(oid)
            try:
                oidx = self.db_valid.get_index_from_ID(oid_i)
            except ValueError:
                continue
            off = self.db_valid.individuals[oidx]
            if not off.valid_evaluation:
                continue
            ys_row = self.db_valid.get_unified_objectives(scale=True, index_list=[oidx])[0]
            MOEAD.update_ideal(self._ideal, ys_row)
            self._replace_subproblem(int(k), oid_i)


    def generate_candidate_individuals(self) -> None:
        '''
        Apply replacements for the previous generation's evaluated offspring, then
        generate this generation's MOEA/D candidates into `db_candidate`.
        '''
        if self._pending:
            self._ensure_state()
            assert self._ideal is not None and self._slot_ids is not None
            self._apply_pending_replacements()
            self._pending.clear()

        self._ensure_state()
        assert self._ideal is not None and self._slot_ids is not None

        mute_rate = (
            self.algorithm_settings.mut_rate
            / max(self.problem.n_input, 1))
        
        if self.db_valid.size <= max(5, int(self.population_size * 0.5)):
            _db = self.db_total
        else:
            _db = self.db_valid
        
        MOEAD.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            ref_dirs=self._ref_dirs,
            neighbors=self._neighbors,
            slot_ids=self._slot_ids,
            prob_neighbor=self.algorithm_settings.prob_neighbor_mating,
            decomposition_method=self._decomposition,
            pbi_theta=self.algorithm_settings.pbi_theta,
            ideal=self._ideal,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.algorithm_settings.pow_poly,
            rng=self._rng,
            pending_list=self._pending,
        )

    def select_elite_from_valid(self) -> None:
        DominanceBasedAlgorithm.select_elite_from_valid(
            self.db_valid, self.db_elite)
