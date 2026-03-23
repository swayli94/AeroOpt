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

from typing import List, Optional, Sequence, Tuple

import numpy as np

from AeroOpt.core import (
    Problem,
    Individual,
    Database,
    MultiProcessEvaluation,
)

from AeroOpt.optimization.moea import DominanceBasedAlgorithm
from AeroOpt.optimization.utils import (
    polynomial_mutation,
    sbx_crossover,
)
from AeroOpt.optimization.base import OptBaseFramework
from AeroOpt.optimization.stochastic.nsgaiii import NSGAIII
from AeroOpt.optimization.settings import (
    SettingsMOEAD,
    SettingsOptimization,
)


class MOEAD(object):
    '''
    MOEA/D operators: weights, neighborhoods, decomposition, mating and
    subproblem replacement (pymoo-style).
    '''

    @staticmethod
    def neighbor_indices(
            ref_dirs: np.ndarray,
            n_neighbors: int,
            ) -> np.ndarray:
        '''
        功能：按权重向量之间的欧氏距离，为每个子问题选取最近的若干邻居下标。

        输入：
            ref_dirs — 权重矩阵，形状 [N, M]。
            n_neighbors — 每个子问题保留的邻居个数（含自身）。

        输出：
            neighbors — 整型数组，形状 [N, n_neighbors]，第 k 行为与第 k 个
                权重最近的 n_neighbors 个子问题下标（升序距离）。
        '''
        w = np.asarray(ref_dirs, dtype=float)
        n = w.shape[0]
        t = max(1, min(int(n_neighbors), n))
        diff = w[:, None, :] - w[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return np.argsort(dist, axis=1, kind='stable')[:, :t]

    @staticmethod
    def default_decomposition_name(n_objective: int) -> str:
        '''
        功能：与 pymoo 默认一致，≤2 目标用 Tchebycheff，否则用 PBI。

        输入：n_objective — 目标个数 M。

        输出：'tchebicheff' 或 'pbi'。
        '''
        return 'tchebicheff' if int(n_objective) <= 2 else 'pbi'

    @staticmethod
    def decomposed_values(
            F: np.ndarray,
            weights: np.ndarray,
            ideal: np.ndarray,
            method: str,
            pbi_theta: float,
            ) -> np.ndarray:
        '''
        功能：计算一组解在给定权重与理想点下的标量适应值（越小越好）。

        输入：
            F — 目标矩阵 [n, M]（已与框架一致，最小化）。
            weights — 权重矩阵 [n, M]，每行一条权重（非负即可，内部会归一化）。
            ideal — 理想点，形状 [M]。
            method — 'tchebicheff' 或 'pbi'。
            pbi_theta — PBI 的惩罚系数 θ。

        输出：长度 n 的一维 float 数组，各解的分解适应值。
        '''
        F = np.asarray(F, dtype=float)
        lam = np.asarray(weights, dtype=float)
        z = np.asarray(ideal, dtype=float)
        n, m = F.shape
        lam_n = np.maximum(lam, 1.0e-32)
        row_norm = np.linalg.norm(lam_n, axis=1, keepdims=True)
        row_norm = np.maximum(row_norm, 1.0e-32)
        lam_unit = lam_n / row_norm

        diff = F - z[None, :]

        if method == 'tchebicheff':
            return np.max(lam_n * np.abs(diff), axis=1)

        if method == 'pbi':
            d1 = np.sum(diff * lam_unit, axis=1)
            d1 = np.maximum(d1, 0.0)
            norm_l = np.linalg.norm(lam_unit, axis=1)
            norm_l = np.maximum(norm_l, 1.0e-32)
            proj = (d1 / norm_l)[:, None] * lam_unit
            d2 = np.linalg.norm(diff - proj, axis=1)
            return d1 + float(pbi_theta) * d2

        raise ValueError(f'Unknown decomposition method: {method}')

    @staticmethod
    def select_parent_slots(
            k: int,
            neighbors: np.ndarray,
            n_pop: int,
            n_parents: int,
            prob_neighbor: float,
            rng: np.random.Generator,
            ) -> np.ndarray:
        '''
        功能：为子问题 k 选择用于交叉的父代槽位下标（指向当前种群中的子问题下标）。

        输入：
            k — 当前子问题下标。
            neighbors — 邻居矩阵，形状 [N, T]，行为各子问题的邻居子问题下标。
            n_pop — 子问题个数 N。
            n_parents — 父代个数（一般为 2）。
            prob_neighbor — 以邻居池选父代的概率。
            rng — NumPy 随机数生成器。

        输出：长度 n_parents 的一维 int 数组，父代对应的子问题下标。
        '''
        if rng.random() < prob_neighbor:
            pool = neighbors[k, :]
            return rng.choice(pool, size=n_parents, replace=False)
        return rng.choice(n_pop, size=n_parents, replace=False)

    @staticmethod
    def objectives_matrix_for_ids(
            db: Database,
            id_list: Sequence[int],
            ) -> np.ndarray:
        '''
        功能：按个体 ID 从数据库取出目标向量，并做与 NSGA-III 相同的最小化统一变换。

        输入：
            db — 数据库（通常为 db_valid）。
            id_list — 个体 ID 序列，长度 n。

        输出：形状 [n, M] 的 float 数组。
        '''
        idx = [db.get_index_from_ID(int(i)) for i in id_list]
        return NSGAIII._unified_objectives_matrix(db, idx)

    @staticmethod
    def replace_subproblem(
            db: Database,
            slot_ids: np.ndarray,
            neighbors: np.ndarray,
            ref_dirs: np.ndarray,
            ideal: np.ndarray,
            k: int,
            offspring_id: int,
            method: str,
            pbi_theta: float,
            ) -> np.ndarray:
        '''
        功能：用子代更新子问题 k 的邻居中分解值更优的槽位（MOEA/D 环境选择）。

        输入：
            db — 可行解数据库。
            slot_ids — 长度 N 的 ID 数组，slot_ids[j] 为子问题 j 当前代表解 ID。
            neighbors — 邻居下标矩阵 [N, T]。
            ref_dirs — 权重 [N, M]。
            ideal — 当前理想点 [M]。
            k — 当前产生子代的子问题下标。
            offspring_id — 子代个体 ID（须已在 db 中且已评估）。
            method、pbi_theta — 分解类型与 PBI 参数。

        输出：更新后的 slot_ids（与输入共享同一数组则原地修改；返回该数组引用）。
        '''
        oidx = db.get_index_from_ID(int(offspring_id))
        off = db.individuals[oidx]
        if not off.valid_evaluation:
            return slot_ids

        Nloc = neighbors[k, :]
        ids = slot_ids[Nloc]
        F_nei = MOEAD.objectives_matrix_for_ids(db, ids)
        F_off = MOEAD.objectives_matrix_for_ids(db, [offspring_id])
        w_nei = ref_dirs[Nloc, :]

        fv_nei = MOEAD.decomposed_values(
            F_nei, w_nei, ideal, method, pbi_theta)
        fv_off = MOEAD.decomposed_values(
            F_off, w_nei, ideal, method, pbi_theta)

        better = np.where(fv_off < fv_nei)[0]
        for j in better:
            slot_ids[Nloc[int(j)]] = int(offspring_id)
        return slot_ids

    @staticmethod
    def update_ideal(
            ideal: np.ndarray,
            F_row: np.ndarray,
            ) -> np.ndarray:
        '''
        功能：用单个个体的目标向量更新逐维理想点（分量最小值）。

        输入：
            ideal — 当前理想点 [M]。
            F_row — 单个目标向量 [M]。

        输出：更新后的 ideal（原地写入并返回）。
        '''
        F_row = np.asarray(F_row, dtype=float)
        np.minimum(ideal, F_row, out=ideal)
        return ideal

    @staticmethod
    def generate_candidate_individuals(
            db_valid: Database,
            db_candidate: Database,
            ref_dirs: np.ndarray,
            neighbors: np.ndarray,
            slot_ids: np.ndarray,
            prob_neighbor: float,
            decomposition_method: str,
            pbi_theta: float,
            ideal: np.ndarray,
            iteration: int,
            cross_rate: float,
            pow_sbx: float,
            mut_rate: float,
            pow_poly: float,
            rng: np.random.Generator,
            pending_list: List[Tuple[int, int]],
            ) -> None:
        '''
        功能：按 MOEA/D 并行子代策略生成一整代子代写入 db_candidate，并记录
            (子问题下标, 子代 ID) 供评估后做邻居替换。

        输入：
            db_valid — 可行归档（用于按 ID 取父代 x）。
            db_candidate — 清空后写入子代。
            ref_dirs, neighbors — 权重与邻居结构。
            slot_ids — 各子问题当前代表解 ID。
            prob_neighbor — 邻域交配概率。
            decomposition_method, pbi_theta — 分解方法（此处仅由上层在替换阶段使用）。
            ideal — 理想点（传入以保持签名一致；本函数不修改）。
            iteration — 当前迭代号。
            cross_rate, pow_sbx, mut_rate, pow_poly — SBX/PM 参数。
            rng — 随机数生成器。
            pending_list — 空列表，函数内填入 (k, offspring_ID)。

        输出：无；副作用为填充 db_candidate 与 pending_list。
        '''
        _ = decomposition_method, pbi_theta, ideal

        n_pop = int(ref_dirs.shape[0])
        if db_valid.size <= 0:
            raise RuntimeError("No valid individuals available for MOEA/D.")

        db_candidate.empty_database()
        pending_list.clear()

        order = rng.permutation(n_pop)
        n_parents = 2

        for k in order:
            p_slots = MOEAD.select_parent_slots(
                int(k), neighbors, n_pop, n_parents,
                prob_neighbor, rng)
            id1 = int(slot_ids[p_slots[0]])
            id2 = int(slot_ids[p_slots[1]])
            i1 = db_valid.get_index_from_ID(id1)
            i2 = db_valid.get_index_from_ID(id2)
            x1, x2 = sbx_crossover(
                db_valid.individuals[i1].x,
                db_valid.individuals[i2].x,
                problem=db_candidate.problem,
                cross_rate=cross_rate, pow_sbx=pow_sbx)
            pick = x1 if rng.random() < 0.5 else x2
            pick = polynomial_mutation(
                pick, problem=db_candidate.problem,
                mut_rate=mut_rate, pow_poly=pow_poly)

            indi = Individual(problem=db_candidate.problem, x=pick)
            indi.source = 'MOEAD'
            indi.generation = int(iteration)
            added, _ = db_candidate.add_individual(
                indi, check_duplication=True, check_bounds=True,
                deepcopy=False, print_warning_info=False)
            if not added:
                continue
            pending_list.append((int(k), int(indi.ID)))

    @staticmethod
    def apply_pending_replacements(
            db: Database,
            pending: Sequence[Tuple[int, int]],
            slot_ids: np.ndarray,
            neighbors: np.ndarray,
            ref_dirs: np.ndarray,
            ideal: np.ndarray,
            method: str,
            pbi_theta: float,
            ) -> None:
        '''
        功能：按生成顺序依次用已评估子代更新理想点并执行邻居替换。

        输入：
            db — 评估后的可行归档（含新子代）。
            pending — (子问题下标 k, 子代 ID) 列表，顺序须与生成时一致。
            slot_ids, neighbors, ref_dirs, ideal — MOEA/D 状态（ideal 与 slot_ids 会被更新）。
            method, pbi_theta — 分解参数。

        输出：无。

        若子代未进入可行归档（例如违反约束），则跳过该条 pending。
        '''
        for k, oid in pending:
            oid_i = int(oid)
            try:
                oidx = db.get_index_from_ID(oid_i)
            except ValueError:
                continue
            off = db.individuals[oidx]
            if not off.valid_evaluation:
                continue
            F_row = NSGAIII._unified_objectives_matrix(db, [oidx])[0]
            MOEAD.update_ideal(ideal, F_row)
            MOEAD.replace_subproblem(
                db, slot_ids, neighbors, ref_dirs,
                ideal, int(k), oid_i, method, pbi_theta)


class OptMOEAD(OptBaseFramework):
    '''
    MOEA/D 与 OptBaseFramework 的对接：维护子问题槽位、理想点与待替换队列。
    '''

    def __init__(self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsMOEAD,
            user_func=None,
            mp_evaluation: MultiProcessEvaluation = None,
            ):

        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            user_func=user_func,
            mp_evaluation=mp_evaluation)

        self.algorithm_settings = algorithm_settings
        n_obj = int(self.problem.n_objective)
        if n_obj < 2:
            raise ValueError("MOEA/D requires at least two objectives.")

        p = self.algorithm_settings.n_partitions
        if p is None:
            p = NSGAIII.suggest_n_partitions(
                n_obj, self.population_size)
        ref = NSGAIII.das_dennis_reference_points(n_obj, p)
        if ref.shape[0] != self.population_size:
            raise ValueError(
                "MOEA/D: population_size must equal the number of Das–Dennis "
                f"reference points ({ref.shape[0]}) for n_partitions={p}. "
                "Adjust population_size or n_partitions in settings.")

        self._ref_dirs = np.asarray(ref, dtype=float)
        t = max(1, min(self.algorithm_settings.n_neighbors, self._ref_dirs.shape[0]))
        self._neighbors = MOEAD.neighbor_indices(self._ref_dirs, t)

        dec = self.algorithm_settings.decomposition
        if dec == 'auto':
            dec = MOEAD.default_decomposition_name(n_obj)
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
        功能：调用基类主循环，并在结束后对最后一代尚未应用的子代执行邻居替换，
            与 pymoo 逐个子代更新在代数末尾的状态对齐。

        输入/输出：无；副作用为可能更新 ``_slot_ids``、``_ideal`` 并清空 ``_pending``。
        '''
        super().main()
        if self._pending:
            self._ensure_state()
            assert self._ideal is not None and self._slot_ids is not None
            MOEAD.apply_pending_replacements(
                self.db_valid, self._pending, self._slot_ids,
                self._neighbors, self._ref_dirs, self._ideal,
                self._decomposition, self.algorithm_settings.pbi_theta)
            self._pending.clear()

    def _ensure_state(self) -> None:
        '''
        功能：在首次进化迭代前，为 N 个子问题槽位绑定可行代表个体 ID，并据其目标计算理想点。

        可行解个数 n_valid 可以小于 N：此时按下标 i % n_valid 循环复用归档中的可行个体，
        使每个权重仍有一个“槽位”（可能暂时对应同一决策向量），与经典实现里“每子问题一解”
        在代数推进后由邻域替换逐步拉开；仅当没有可行解时报错。

        输入/输出：仅操作 ``self._slot_ids``、``self._ideal``；无参数与返回值。
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
        self._ideal = MOEAD.objectives_matrix_for_ids(
            self.db_valid, self._slot_ids).min(axis=0).astype(float)

    def generate_candidate_individuals(self) -> None:
        '''
        功能：先处理上一轮已评估子代的替换，再生成本轮 MOEA/D 子代。

        输入/输出：遵循基类约定，写入 ``db_candidate``；无显式参数与返回值。
        '''
        if self._pending:
            self._ensure_state()
            assert self._ideal is not None and self._slot_ids is not None
            MOEAD.apply_pending_replacements(
                self.db_valid, self._pending, self._slot_ids,
                self._neighbors, self._ref_dirs, self._ideal,
                self._decomposition, self.algorithm_settings.pbi_theta)
            self._pending.clear()

        self._ensure_state()
        assert self._ideal is not None and self._slot_ids is not None

        mute_rate = (
            self.algorithm_settings.mut_rate
            / max(self.problem.n_input, 1))

        MOEAD.generate_candidate_individuals(
            db_valid=self.db_valid,
            db_candidate=self.db_candidate,
            ref_dirs=self._ref_dirs,
            neighbors=self._neighbors,
            slot_ids=self._slot_ids,
            prob_neighbor=self.algorithm_settings.prob_neighbor_mating,
            decomposition_method=self._decomposition,
            pbi_theta=self.algorithm_settings.pbi_theta,
            ideal=self._ideal,
            iteration=self.iteration,
            cross_rate=self.algorithm_settings.cross_rate,
            pow_sbx=self.algorithm_settings.pow_sbx,
            mut_rate=mute_rate,
            pow_poly=self.algorithm_settings.pow_poly,
            rng=self._rng,
            pending_list=self._pending,
        )

    def select_elite_from_valid(self) -> None:
        '''
        功能：从可行归档中选精英集（非支配排序 + 拥挤距离，与 NSGA-II 一致）。

        输入/输出：同 ``DominanceBasedAlgorithm.select_elite_from_valid``。
        '''
        DominanceBasedAlgorithm.select_elite_from_valid(
            self.db_valid, self.db_elite)
