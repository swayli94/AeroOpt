'''
RVEA: Reference Vector Guided Evolutionary Algorithm

In RVEA, a scalarization approach, termed angle penalized distance (APD), is adopted to 
balance the convergence and diversity of the solutions in the high-dimensional objective space.
Furthermore, an adaptation strategy is proposed to dynamically adjust the 
reference vectors' distribution according to the objective functions' scales. 

Note that the APD is adapted based on the progress the algorithm has made.
Thus, termination criteria such as n_gen or n_evals should be used.

References:
    
    Ran Cheng, Yaochu Jin, Markus Olhofer, and Bernhard Sendhoff.
    A reference vector guided evolutionary algorithm for many-objective optimization.
    IEEE Transactions on Evolutionary Computation, 20(5):773-791, 2016. doi:10.1109/TEVC.2016.2519378.
    
    https://pymoo.org/algorithms/moo/rvea.html#nb-rvea
    
    https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/rvea.py
    
'''

from __future__ import annotations

from typing import List, Optional, Tuple, Callable

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
    binary_tournament_selection,
    polynomial_mutation,
    sbx_crossover,
)
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.settings import (
    SettingsRVEA,
    SettingsOptimization,
)


class RVEAApdState(object):
    '''
    Mutable state for APD survival: running ideal, nadir, and adapted unit
    reference directions V (pymoo RVEA / APDSurvival).
    '''

    def __init__(self, ref_dirs: np.ndarray) -> None:
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        n_dim = int(ref_dirs.shape[1])
        self.ref_dirs_base = ref_dirs
        self.ideal = np.full(n_dim, np.inf, dtype=float)
        self.nadir: Optional[np.ndarray] = None
        self.V = self._calc_unit_ref_dirs(self.ref_dirs_base)
        self.gamma = self._calc_reference_gamma(self.V)

    def adapt(self) -> None:
        if self.nadir is None:
            return
        span = self.nadir - self.ideal
        span = np.maximum(span, 1.0e-64)
        self.V = self._calc_adapted_unit_ref_dirs(
            self.ref_dirs_base, span)
        self.gamma = self._calc_reference_gamma(self.V)

    @staticmethod
    def _calc_unit_ref_dirs(ref_dirs: np.ndarray) -> np.ndarray:
        '''
        Normalize reference directions row-wise to unit vectors.
        '''
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1.0e-64)
        return ref_dirs / norms

    @staticmethod
    def _calc_adapted_unit_ref_dirs(ref_dirs: np.ndarray, span: np.ndarray) -> np.ndarray:
        '''
        Adapt reference directions to the scale of the objective space and normalize them.

        This function first normalizes the input reference directions to unit vectors,
        then rescales each dimension according to the provided `span` (typically the
        range of each objective), and finally normalizes the resulting vectors again.

        The purpose is to adjust the reference directions so that they are consistent
        with the anisotropic scaling of the objective space, preventing objectives with
        larger numerical ranges from dominating the decomposition.
        
        The nadir point is the vector composed of the worst (maximum, for minimization problems)
        objective values among all solutions on the Pareto front.

        Parameters
        ----------
        ref_dirs : np.ndarray [n_subproblems, n_objectives]
            Array of reference directions (weight vectors).
        span : np.ndarray [n_objective]
            Scaling factor for each objective (e.g., the difference between nadir and
            ideal points). Used to adapt the reference directions to the actual
            objective space.

        Returns
        -------
        adapted_ref_dirs: np.ndarray [n_subproblems, n_objectives]
            Adapted reference directions, normalized to unit vectors after scaling.
        '''
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        span = np.asarray(span, dtype=float)
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1.0e-64)
        ref_dirs = ref_dirs / norms * span[None, :]
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1.0e-64)
        return ref_dirs / norms

    @staticmethod
    def _calc_reference_gamma(V: np.ndarray) -> np.ndarray:
        '''
        Compute minimal inter-vector angle (except self) for each reference vector.
        '''
        V = np.asarray(V, dtype=float)
        n = V.shape[0]
        if n <= 1:
            return np.array([1.0e-64], dtype=float)
        cosines = V @ V.T
        cosines = np.clip(cosines, -1.0, 1.0)
        sorted_neg = np.sort(-cosines, axis=1)
        second_cos = -sorted_neg[:, 1]
        second_cos = np.clip(second_cos, -1.0, 1.0)
        gamma = np.arccos(second_cos)
        return np.maximum(gamma, 1.0e-64)


class RVEA(Algorithm):
    '''
    RVEA operators (APD environmental selection, Das-Dennis reference directions).
    '''

    @staticmethod
    def environmental_selection_indices(
            db: Database,
            population_size: int,
            state: RVEAApdState|None = None,
            iteration: int = 0,
            max_iterations: int = 20,
            alpha: float = 2.0
            ) -> List[int]:
        '''
        Select up to one survivor per reference direction via APD (pymoo APDSurvival).
        Returns local indices into `db.individuals`.
        
        Parameters
        ----------
        db: Database
            Database to select from.
        population_size: int
            Population size.
        state: RVEAApdState
            State of the RVEA algorithm.
        iteration: int
            Current iteration.
        max_iterations: int
            Maximum number of iterations.
        alpha: float
            APD penalty parameter amplifies the progress (0 to 1) of the algorithm.
            In the early stages of the algorithm, the penalty is small, allowing for more convergence.
            In the later stages of the algorithm, the penalty is large, allowing for more diversity.
            Therefore, a larger `alpha` value will lead to more convergence.
            
        Returns
        -------
        indices: List[int]
            Indices of the selected individuals.
        '''
        if db.size <= 0:
            return []
        n_obj = db.problem.n_objective
        if n_obj <= 1:
            if db.size <= population_size:
                return list(range(db.size))
            DominanceBasedAlgorithm.non_dominated_ranking(db)
            DominanceBasedAlgorithm.assign_crowding_distance(db)
            return DominanceBasedAlgorithm.select_parent_indices(db, population_size)

        if db.size <= population_size:
            return list(range(db.size))

        ys = db.get_unified_objectives(scale=True)
        
        if state is None:
            raise ValueError("State is required for APD environmental selection.")

        state.ideal = np.minimum(ys.min(axis=0), state.ideal)
        F_shift = ys - state.ideal

        dist_to_ideal = np.linalg.norm(F_shift, axis=1)
        dist_to_ideal = np.maximum(dist_to_ideal, 1.0e-64)
        F_prime = F_shift / dist_to_ideal[:, None]

        V = state.V
        cos_fp = F_prime @ V.T
        cos_fp = np.clip(cos_fp, -1.0, 1.0)
        acute_angle = np.arccos(cos_fp)
        niches = np.argmin(acute_angle, axis=1)

        niches_to_ind: List[List[int]] = [[] for _ in range(V.shape[0])]
        for ind_row, niche_j in enumerate(niches):
            niches_to_ind[int(niche_j)].append(int(ind_row))

        n_max_gen = max(int(max_iterations), 1)
        n_gen = max(int(iteration), 1)
        M = float(n_obj) if n_obj > 2 else 1.0
        progress = min(1.0, n_gen / float(n_max_gen))
        
        '''
        `progress` is the progress of the algorithm, normalized to the number of iterations.
        `alpha` is the APD penalty parameter that controls the trade-off between convergence and diversity.
        '''

        survivor_records: List[Tuple[int, int, float]] = []

        for k in range(V.shape[0]):
            assigned = niches_to_ind[k]
            if len(assigned) == 0:
                continue
            gamma_k = float(state.gamma[k])
            theta = acute_angle[assigned, k]
            penalty = M * (progress ** float(alpha)) * (theta / gamma_k)
            apd = dist_to_ideal[assigned] * (1.0 + penalty)
            j_local = int(np.argmin(apd))
            survivor = int(assigned[j_local])
            survivor_records.append((survivor, k, float(apd[j_local])))

        if not survivor_records:
            DominanceBasedAlgorithm.non_dominated_ranking(db)
            DominanceBasedAlgorithm.assign_crowding_distance(db)
            return DominanceBasedAlgorithm.select_parent_indices(db, population_size)

        survivor_records.sort(key=lambda t: t[2])
        survivors = [t[0] for t in survivor_records]

        if len(survivors) > population_size:
            survivors = survivors[:population_size]
        elif len(survivors) < population_size:
            taken = set(survivors)
            rest = [i for i in range(db.size) if i not in taken]
            dist_rest = dist_to_ideal[rest]
            order = np.argsort(dist_rest)
            for j in order:
                if len(survivors) >= population_size:
                    break
                survivors.append(int(rest[int(j)]))

        ys_surv = ys[survivors, :]
        state.nadir = ys_surv.max(axis=0)

        return survivors

    @staticmethod
    def build_temporary_parent_database(
            db: Database,
            population_size: int,
            state: RVEAApdState|None = None,
            iteration: int = 0,
            max_iterations: int = 20,
            alpha: float = 2.0
            ) -> Database:
        '''
        Temporary parent pool from the population database via RVEA APD truncation.
        Reference directions are taken from `state` (set in `OptRVEA` init).
        '''
        if db.size <= 0:
            raise ValueError("Cannot build parent database from an empty database.")

        db_work = db.get_sub_database(
            index_list=list(range(db.size)), deepcopy=True)
        if db_work.size <= population_size:
            return db_work

        idx = RVEA.environmental_selection_indices(
            db_work, population_size, state,
            iteration, max_iterations, alpha)
        return db_work.get_sub_database(index_list=idx, deepcopy=True)

    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            state: RVEAApdState|None = None,
            max_iterations: int = 20,
            alpha: float = 2.0,
            cross_rate: float = 1.0,
            pow_sbx: float = 20.0,
            mut_rate: float = 1.0,
            pow_poly: float = 20.0,
            ) -> None:
        if db.size <= 0:
            raise RuntimeError("No individuals available for RVEA evolution.")

        temp_parents = RVEA.build_temporary_parent_database(
            db, population_size, state, iteration, max_iterations,
            alpha)
        mating_population = binary_tournament_selection(
            pool=temp_parents, n_select=population_size)

        db_candidate.empty_database()
        n_pairs = int(np.ceil(population_size / 2))
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = sbx_crossover(
                p1.x, p2.x, problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx)

            x1 = polynomial_mutation(
                x1, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly)
            x2 = polynomial_mutation(
                x2, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly)

            for x_child in (x1, x2):
                if db_candidate.size >= population_size:
                    break
                indi = Individual(problem=db_candidate.problem, x=x_child)
                indi.source = 'evolutionary_operator'
                indi.generation = iteration
                db_candidate.add_individual(
                    indi, check_duplication=True, check_bounds=True,
                    deepcopy=False, print_warning_info=False)


class OptRVEA(OptBaseFramework):
    '''
    RVEA optimization (APD truncation for the mating pool, reference-vector adaptation).
    '''
    def __init__(self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsRVEA,
            user_func: Callable|None = None,
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
        n_obj = self.problem.n_objective
        p = self.algorithm_settings.n_partitions
        if p is None:
            p = DecompositionBasedAlgorithm.suggest_n_partitions(
                    n_obj, self.population_size)
        ref_pts = DecompositionBasedAlgorithm.das_dennis_reference_points(n_obj, p)
        self._apd_state = RVEAApdState(ref_pts)

    def update_parameters(self) -> None:
        af = self.algorithm_settings.adapt_freq
        if af is None or af <= 0:
            return None
        period = max(1, int(np.ceil(self.max_iterations * float(af))))
        if self.iteration > 0 and self.iteration % period == 0:
            self._apd_state.adapt()
        return None

    def generate_candidate_individuals(self) -> None:
        mute_rate = (
            self.algorithm_settings.mut_rate
            / max(self.problem.n_input, 1))

        if self.db_valid.size <= max(5, int(self.population_size * 0.5)):
            _db = self.db_total
        else:
            _db = self.db_valid

        RVEA.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            state=self._apd_state,
            max_iterations=self.max_iterations,
            alpha=self.algorithm_settings.alpha,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.algorithm_settings.pow_poly,
        )

    def select_elite_from_valid(self) -> None:
        DominanceBasedAlgorithm.select_elite_from_valid(
            self.db_valid, self.db_elite)
