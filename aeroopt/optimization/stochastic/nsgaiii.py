'''
NSGA-III implementation (reference-point-based environmental selection).

Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, part I: solving problems
with box constraints. IEEE TEC, 18(4), 577-601.
'''

from __future__ import annotations

from typing import List, Optional, Callable

import numpy as np

from aeroopt.core import (
    Problem,
    Individual,
    Database,
    MultiProcessEvaluation,
)

from aeroopt.optimization.moea import (
    Algorithm, DominanceBasedAlgorithm, DecompositionBasedAlgorithm
)
from aeroopt.optimization.utils import (
    associate_to_reference,
    binary_tournament_selection,
    polynomial_mutation,
    reference_directions,
    sbx_crossover,
)
from aeroopt.optimization.base import OptBaseFramework
from aeroopt.optimization.settings import (
    SettingsNSGAIII, SettingsOptimization
)


class NSGAIII(Algorithm):
    '''
    NSGA-III operators.
    '''

    @staticmethod
    def _normalize_objectives_nsgaiii(Z: np.ndarray) -> np.ndarray:
        '''
        Ideal-point shift and intercept scaling (NSGA-III style).
        '''
        eps = 1.0e-12
        _, m = Z.shape
        z_min = Z.min(axis=0)
        zp = Z - z_min
        if m == 1:
            return zp / np.maximum(zp.max(), eps)

        extreme = np.zeros((m, m), dtype=float)
        for i in range(m):
            w = np.full(m, 1.0e-6)
            w[i] = 1.0
            asf = np.max(zp / w, axis=1)
            k = int(np.argmin(asf))
            extreme[i, :] = zp[k, :]
        try:
            plane = np.linalg.solve(extreme, np.ones(m))
            intercepts = 1.0 / np.maximum(plane, eps)
        except np.linalg.LinAlgError:
            intercepts = np.maximum(zp.max(axis=0), eps)
        intercepts = np.maximum(intercepts, eps)
        return zp / intercepts

    @staticmethod
    def _select_population_indices_nsgaiii(
            db: Database,
            index_fronts: List[List[int]],
            population_size: int,
            ref_points: np.ndarray,
            ) -> List[int]:
        '''
        Fill `population_size` slots using fast non-dominated sorting fronts,
        then reference-point niching on the last partial front.
        '''
        n_obj = db.problem.n_objective
        if n_obj <= 1 or ref_points.size == 0:
            return DominanceBasedAlgorithm.select_parent_indices(
                db, population_size)

        selected: List[int] = []
        for front in index_fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
                if len(selected) == population_size:
                    return selected
                continue

            k_need = population_size - len(selected)
            union_idx = selected + list(front)
            z_union = db.get_unified_objectives(scale=True, index_list=union_idx)
            zn = NSGAIII._normalize_objectives_nsgaiii(z_union)
            row_of = {g: r for r, g in enumerate(union_idx)}

            ref_dirs = reference_directions(ref_points)

            niche = np.zeros(ref_dirs.shape[0], dtype=int)
            for idx in selected:
                z = zn[row_of[idx], :]
                j, _ = associate_to_reference(z, ref_dirs)
                niche[j] += 1

            pool = list(front)
            assoc_j = np.zeros(len(pool), dtype=int)
            assoc_d = np.zeros(len(pool), dtype=float)
            for pi, idx in enumerate(pool):
                z = zn[row_of[idx], :]
                j, d = associate_to_reference(z, ref_dirs)
                assoc_j[pi] = j
                assoc_d[pi] = d

            picked: List[int] = []
            for _ in range(k_need):
                if not pool:
                    break
                rho_min = float(np.min(niche))
                refs_in_pool = {int(assoc_j[pi]) for pi in range(len(pool))}
                j_star_set = [j for j in refs_in_pool if niche[j] == rho_min]
                if not j_star_set:
                    j_star_set = list(refs_in_pool)

                best_pi = None
                best_key = None
                for pi in range(len(pool)):
                    j = int(assoc_j[pi])
                    if j not in j_star_set:
                        continue
                    key = (float(assoc_d[pi]), int(pool[pi]))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pi = pi
                if best_pi is None:
                    break
                j_pick = int(assoc_j[best_pi])
                idx_take = pool.pop(best_pi)
                niche[j_pick] += 1
                assoc_j = np.delete(assoc_j, best_pi)
                assoc_d = np.delete(assoc_d, best_pi)
                picked.append(idx_take)

            selected.extend(picked)
            break

        return selected

    @staticmethod
    def environmental_selection_indices(
            db: Database,
            population_size: int,
            n_partitions: int = 1,
            ) -> List[int]:
        '''
        Indices of individuals to keep (length <= population_size).
        '''
        if db.size <= 0:
            return []
        index_fronts = DominanceBasedAlgorithm.non_dominated_ranking(db)
        n_obj = db.problem.n_objective
        p = max(1, int(n_partitions))
        ref_pts = DecompositionBasedAlgorithm.das_dennis_reference_points(n_obj, p)
        return NSGAIII._select_population_indices_nsgaiii(
            db, index_fronts, population_size, ref_pts)

    @staticmethod
    def build_temporary_parent_database(
            db: Database,
            population_size: int,
            n_partitions: Optional[int] = None,
            ) -> Database:
        '''
        Temporary parent pool from the population database via NSGA-III environmental
        selection (reference points). Deep copy; does not modify `db`.
        '''
        if db.size <= 0:
            raise ValueError("Cannot build parent database from an empty archive.")

        db_work = db.get_sub_database(
            index_list=list(range(db.size)), deepcopy=True)
        DominanceBasedAlgorithm.non_dominated_ranking(
            db_work)
        if db_work.size <= population_size:
            DominanceBasedAlgorithm.assign_crowding_distance(db_work)
            return db_work
        p = n_partitions
        if p is None:
            p = DecompositionBasedAlgorithm.suggest_n_partitions(
                db.problem.n_objective, population_size)
        idx = NSGAIII.environmental_selection_indices(db_work, population_size, p)
        return db_work.get_sub_database(index_list=idx, deepcopy=True)

    @staticmethod
    def generate_candidate_individuals(
            db: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            cross_rate: float = 1.0,
            pow_sbx: float = 20.0,
            mut_rate: float = 1.0,
            pow_poly: float = 20.0,
            n_partitions: Optional[int] = None,
            rng: np.random.Generator|None = None,
            ) -> None:
        '''
        Like `NSGAII.generate_candidate_individuals`, but the temporary parent
        pool uses NSGA-III reference-point truncation from `db`.
        '''
        if db.size <= 0:
            raise RuntimeError("No individuals available for NSGA-III evolution.")

        if rng is None:
            rng = np.random.default_rng()

        temp_parents = NSGAIII.build_temporary_parent_database(
            db, population_size, n_partitions=n_partitions)
        mating_population = binary_tournament_selection(
            pool=temp_parents, n_select=population_size, rng=rng)

        db_candidate.empty_database()
        n_pairs = int(np.ceil(population_size / 2))
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = sbx_crossover(
                p1.x, p2.x, problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx, rng=rng)

            x1 = polynomial_mutation(
                x1, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly, rng=rng)
            x2 = polynomial_mutation(
                x2, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly, rng=rng)

            for x_child in (x1, x2):
                if db_candidate.size >= population_size:
                    break
                indi = Individual(problem=db_candidate.problem, x=x_child)
                indi.source = 'evolutionary_operator'
                indi.generation = iteration
                db_candidate.add_individual(
                    indi, check_duplication=True, check_bounds=True,
                    deepcopy=False, print_warning_info=False)


class OptNSGAIII(OptBaseFramework):
    '''
    NSGA-III optimization (reference-point truncation for the mating pool).

    Parameters:
    -----------
    problem: Problem
        Problem for optimization.
    optimization_settings: SettingsOptimization
        Settings of the optimization.
    algorithm_settings: SettingsNSGAIII
        NSGA-III-specific settings.
    user_func: Callable
        User-defined function to evaluate the individuals.
        If None, use external evaluation script.
    mp_evaluation: MultiProcessEvaluation
        Multi-process evaluation object defined in the entrance of the entire program.
        If None, use serial evaluation.
    rng: np.random.Generator
        Optional NumPy random generator.
    '''

    def __init__(self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsNSGAIII,
            user_func: Callable|None = None,
            user_func_supports_parallel: bool = False,
            mp_evaluation: MultiProcessEvaluation|None = None,
            rng: np.random.Generator|None = None,
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
        self.rng = rng

    def generate_candidate_individuals(self) -> None:
        mute_rate = (
            self.algorithm_settings.mut_rate
            / max(self.problem.n_input, 1))

        if self.db_valid.size <= 0:
            _db = self.db_total
        else:
            _db = self.db_valid

        NSGAIII.generate_candidate_individuals(
            db=_db,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.algorithm_settings.pow_poly,
            n_partitions=self.algorithm_settings.n_partitions,
            rng=self.rng)

    def select_elite_from_valid(self) -> None:
        '''
        Select elite individuals from the valid database.
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(self.db_valid, self.db_elite)
