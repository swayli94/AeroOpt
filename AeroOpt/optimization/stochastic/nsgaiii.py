'''
NSGA-III implementation (reference-point-based environmental selection).

Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, part I: solving problems
with box constraints. IEEE TEC, 18(4), 577-601.
'''

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from AeroOpt.core import (
    Problem,
    Individual,
    Database,
    MultiProcessEvaluation,
)
from AeroOpt.optimization.stochastic.nsgaii import NSGAII
from AeroOpt.optimization.stochastic.base import (
    OptEvolutionaryFramework,
    EvolutionaryAlgorithm,
)
from AeroOpt.optimization.settings import (
    SettingsNSGAIII, SettingsOptimization
)


class NSGAIII(NSGAII):
    '''
    NSGA-III: same variation operators as NSGA-II; the temporary parent pool
    (see `build_temporary_parent_database`) is filled using reference points
    and niche preservation instead of crowding distance.
    '''

    def __init__(self, settings_name: str = "default",
            fname_settings: str = 'settings.json'):

        EvolutionaryAlgorithm.__init__(self, algorithm_name='NSGAIII')
        self.settings = SettingsNSGAIII(
            name=settings_name, fname_settings=fname_settings)

    @staticmethod
    def suggest_n_partitions(n_objective: int, population_size: int) -> int:
        '''
        Pick a simplex partition count so the number of reference points is near
        `population_size` (combinatorial count C(p+M-1, M-1)).
        '''
        if n_objective <= 1:
            return 1
        best_p, best_d = 1, float('inf')
        for p in range(1, 40):
            n_ref = math.comb(p + n_objective - 1, n_objective - 1)
            d = abs(n_ref - population_size)
            if d < best_d:
                best_d, best_p = d, p
        return max(1, best_p)

    @staticmethod
    def das_dennis_reference_points(n_objective: int, n_partitions: int) -> np.ndarray:
        '''
        Das-Dennis reference points on the (M-1)-simplex, shape [n_ref, M].
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
    def _unified_objectives_matrix(db: Database, indices: List[int]) -> np.ndarray:
        n_obj = db.problem.n_objective
        ys = np.zeros((len(indices), n_obj), dtype=float)
        for r, idx in enumerate(indices):
            ys[r, :] = db.individuals[idx].objectives
        j = 0
        for out_type in db.problem.problem_settings.output_type:
            if abs(out_type) != 1:
                continue
            if out_type == -1:
                ys[:, j] = -ys[:, j]
            j += 1
        return ys

    @staticmethod
    def normalize_objectives_nsgaiii(Z: np.ndarray) -> np.ndarray:
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
    def _perpendicular_distance(z: np.ndarray, direction_unit: np.ndarray) -> float:
        t = float(np.dot(z, direction_unit))
        t = max(t, 0.0)
        return float(np.linalg.norm(z - t * direction_unit))

    @staticmethod
    def _reference_directions(ref_points: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1.0e-12)
        return ref_points / norms

    @staticmethod
    def _associate_to_ref(z: np.ndarray, ref_dirs: np.ndarray) -> tuple:
        best_j, best_d = 0, float('inf')
        for j in range(ref_dirs.shape[0]):
            d = NSGAIII._perpendicular_distance(z, ref_dirs[j, :])
            if d < best_d:
                best_d, best_j = d, j
        return best_j, best_d

    @staticmethod
    def select_population_indices_nsgaiii(
            db: Database,
            fronts: List[List[int]],
            population_size: int,
            ref_points: np.ndarray,
            ) -> List[int]:
        '''
        Fill `population_size` slots using fast non-dominated sorting fronts,
        then reference-point niching on the last partial front.
        '''
        n_obj = db.problem.n_objective
        if n_obj <= 1 or ref_points.size == 0:
            return EvolutionaryAlgorithm.select_population_indices(
                db, fronts, population_size)

        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
                if len(selected) == population_size:
                    return selected
                continue

            k_need = population_size - len(selected)
            union_idx = selected + list(front)
            z_union = NSGAIII._unified_objectives_matrix(db, union_idx)
            zn = NSGAIII.normalize_objectives_nsgaiii(z_union)
            row_of = {g: r for r, g in enumerate(union_idx)}

            ref_dirs = NSGAIII._reference_directions(ref_points)

            niche = np.zeros(ref_dirs.shape[0], dtype=int)
            for idx in selected:
                z = zn[row_of[idx], :]
                j, _ = NSGAIII._associate_to_ref(z, ref_dirs)
                niche[j] += 1

            pool = list(front)
            assoc_j = np.zeros(len(pool), dtype=int)
            assoc_d = np.zeros(len(pool), dtype=float)
            for pi, idx in enumerate(pool):
                z = zn[row_of[idx], :]
                j, d = NSGAIII._associate_to_ref(z, ref_dirs)
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
    def build_temporary_parent_database(
            db_valid: Database,
            population_size: int,
            n_partitions: Optional[int] = None,
            ) -> Database:
        '''
        Temporary parent pool from the valid archive via NSGA-III environmental
        selection (reference points). Deep copy; does not modify `db_valid`.
        '''
        if db_valid.size <= 0:
            raise ValueError("Cannot build parent database from an empty valid archive.")

        db_work = db_valid.get_sub_database(
            index_list=list(range(db_valid.size)), deepcopy=True)
        fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(
            db_work, is_valid_database=True)
        if db_work.size <= population_size:
            EvolutionaryAlgorithm.assign_crowding_distance(db_work, fronts)
            return db_work
        p = n_partitions
        if p is None:
            p = NSGAIII.suggest_n_partitions(
                db_valid.problem.n_objective, population_size)
        idx = NSGAIII.environmental_selection(db_work, population_size, p)
        return db_work.get_sub_database(index_list=idx, deepcopy=True)

    @staticmethod
    def environmental_selection(
            combined: Database,
            population_size: int,
            n_partitions: int,
            ) -> List[int]:
        '''
        Indices of individuals to keep (length <= population_size).
        '''
        if combined.size <= 0:
            return []
        fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(
            combined, is_valid_database=True)
        n_obj = combined.problem.n_objective
        p = max(1, int(n_partitions))
        ref_pts = NSGAIII.das_dennis_reference_points(n_obj, p)
        return NSGAIII.select_population_indices_nsgaiii(
            combined, fronts, population_size, ref_pts)

    @staticmethod
    def generate_candidate_individuals(
            db_valid: Database,
            db_candidate: Database,
            population_size: int,
            iteration: int,
            cross_rate: float = 1.0,
            pow_sbx: float = 20.0,
            mut_rate: float = 1.0,
            pow_poly: float = 20.0,
            n_partitions: Optional[int] = None,
            ) -> None:
        '''
        Like `NSGAII.generate_candidate_individuals`, but the temporary parent
        pool uses NSGA-III reference-point truncation from `db_valid`.
        '''
        if db_valid.size <= 0:
            raise RuntimeError("No valid individuals available for NSGA-III evolution.")

        temp_parents = NSGAIII.build_temporary_parent_database(
            db_valid, population_size, n_partitions=n_partitions)
        mating_population = NSGAII.binary_tournament_selection(
            pool=temp_parents, n_select=population_size)

        db_candidate.empty_database()
        n_pairs = int(np.ceil(population_size / 2))
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = NSGAII.sbx_crossover(
                p1.x, p2.x, problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx)

            x1 = NSGAII.polynomial_mutation(
                x1, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly)
            x2 = NSGAII.polynomial_mutation(
                x2, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly)

            for x_child in (x1, x2):
                if db_candidate.size >= population_size:
                    break
                indi = Individual(problem=db_candidate.problem, x=x_child)
                indi.source = "GA"
                indi.generation = iteration
                db_candidate.add_individual(
                    indi, check_duplication=True, check_bounds=True,
                    deepcopy=False, print_warning_info=False)


class OptNSGAIII(OptEvolutionaryFramework):
    '''
    NSGA-III optimization (reference-point truncation for the mating pool).
    '''

    def __init__(self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            evolutionary_algorithm: NSGAIII,
            user_func=None,
            mp_evaluation: MultiProcessEvaluation = None,
            ):

        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            evolutionary_algorithm=evolutionary_algorithm,
            user_func=user_func,
            mp_evaluation=mp_evaluation)

    def generate_candidate_individuals(self) -> None:
        mute_rate = (
            self.evolutionary_algorithm.settings.mut_rate
            / max(self.problem.n_input, 1))

        NSGAIII.generate_candidate_individuals(
            db_valid=self.db_valid,
            db_candidate=self.db_candidate,
            population_size=self.population_size,
            iteration=self.iteration,
            cross_rate=self.evolutionary_algorithm.settings.cross_rate,
            pow_sbx=self.evolutionary_algorithm.settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.evolutionary_algorithm.settings.pow_poly,
            n_partitions=self.evolutionary_algorithm.settings.n_partitions)
