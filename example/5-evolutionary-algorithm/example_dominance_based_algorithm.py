'''
Example: demonstrate dominance-based MOEA helpers (`DominanceBasedAlgorithm`).

- Create a problem `problem`:
  1) n_input=2, n_output=2, n_constraint=1
  2) x1, x2 in [-1, 1]
  3) y1 = x1^2 + x2^2 (use built-in function `user_func`)
  4) y2 = x1 - x2
  5) constraint1: x1^2 + x2^2 - 0.64 <= 0.0

- Create a database `db` for testing:
  1) generate 100 random input points `xs`
  2) use `xs` to build `db`
  3) evaluate individuals with `db.evaluate_individuals` (serial evaluation)

- Plot / verify:
  `check_pareto_dominance`, `non_dominated_ranking`, `assign_crowding_distance`,
  `select_parent_indices`, `rank_pareto`, and truncation via `select_parent_indices`
  after sorting.
'''

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroopt.core import (
    Database, Individual, Problem, SettingsProblem, SettingsData
)
from aeroopt.optimization import (
    DominanceBasedAlgorithm,
    SettingsOptimization,
)


def user_func(x: np.ndarray, **kwargs):
    y1 = float(x[0] ** 2 + x[1] ** 2)
    y2 = float(x[0] - x[1])
    return True, np.array([y1, y2], dtype=float)


def build_minimal_settings_file(settings_path: Path, work_dir: Path) -> None:
    settings = {
        "data_demo_ea": {
            "type": "SettingsData",
            "name": "demo_data_ea",
            "name_input": ["x1", "x2"],
            "input_low": [-1.0, -1.0],
            "input_upp": [1.0, 1.0],
            "input_precision": [0.0, 0.0],
            "name_output": ["y1", "y2"],
            "output_low": [0.0, -2.0],
            "output_upp": [2.0, 2.0],
            "output_precision": [0.0, 0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "problem_demo_ea": {
            "type": "SettingsProblem",
            "name": "demo_problem_ea",
            "name_data_settings": "demo_data_ea",
            "output_type": ["-1", "1"],
            "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"],
        },
        "opt_demo_ea": {
            "type": "SettingsOptimization",
            "name": "demo_opt_ea",
            "resume": False,
            "population_size": 40,
            "max_iterations": 1,
            "fname_db_total": "db-total.json",
            "fname_db_elite": "db-elite.json",
            "fname_db_population": "db-population.json",
            "fname_db_resume": "db-resume.json",
            "fname_log": "optimization.log",
            "working_directory": str(work_dir),
            "info_level_on_screen": 2,
            "critical_potential_x": 0.2,
        },
    }
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def build_candidate_database(problem: Problem, xs: np.ndarray, id_start: int = 1) -> Database:
    db = Database(problem=problem, database_type="population")
    db.individuals = [
        Individual(problem=problem, x=xs[i, :].copy(), ID=id_start + i)
        for i in range(xs.shape[0])
    ]
    db.update_id_list()
    return db


def clone_database(db: Database) -> Database:
    return db.get_sub_database(index_list=list(range(db.size)), deepcopy=True)


def collect_ranks(db: Database) -> np.ndarray:
    return np.array([indi.pareto_rank for indi in db.individuals], dtype=int)


def collect_crowding(db: Database) -> np.ndarray:
    return np.array([indi.crowding_distance for indi in db.individuals], dtype=float)


def get_objectives_matrix(db: Database) -> np.ndarray:
    n = db.size
    m = db.problem.n_objective
    ys = np.zeros((n, m))
    for i, indi in enumerate(db.individuals):
        ys[i, :] = indi.objectives
    return ys


def verify_faster_valid_flag_on_feasible_subset(db_source: Database) -> None:
    feas_idx = [
        i
        for i, indi in enumerate(db_source.individuals)
        if indi.valid_evaluation and indi.sum_violation is not None and indi.sum_violation <= 0.0
    ]
    if len(feas_idx) < 3:
        return

    sub = db_source.get_sub_database(index_list=feas_idx, deepcopy=True)

    d_obj = clone_database(sub)
    d_obj._is_valid_database = True
    DominanceBasedAlgorithm.non_dominated_ranking(d_obj)
    r_obj = collect_ranks(d_obj)

    d_indi = clone_database(sub)
    d_indi._is_valid_database = False
    DominanceBasedAlgorithm.non_dominated_ranking(d_indi)
    r_indi = collect_ranks(d_indi)

    if not np.array_equal(r_obj, r_indi):
        raise AssertionError(
            "On feasible individuals, ranking with scaled objectives (valid DB) "
            "should match Individual.check_dominance (non-valid DB flag)."
        )


def demo_pareto_dominance_print(problem: Problem) -> None:
    pairs = [
        (np.array([0.2, 0.0]), np.array([0.4, 0.6]), "trade-off: better y1 vs better y2 -> 9"),
        (np.array([0.3, 0.2]), np.array([0.5, 0.1]), "unified: A better on every objective -> 1"),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), "identical -> 0"),
    ]
    print("\n[pareto_dominance] unified objective vectors (min y1 -> negate y1; max y2 -> keep y2):")
    for ya, yb, note in pairs:
        ua = _db_from_y_row(problem, ya).get_unified_objectives(scale=True)[0, :]
        ub = _db_from_y_row(problem, yb).get_unified_objectives(scale=True)[0, :]
        flag = DominanceBasedAlgorithm.check_pareto_dominance(ua, ub)
        print(f"  u_a={ua} vs u_b={ub}  -> {flag}  ({note})")


def _db_from_y_row(problem: Problem, y: np.ndarray) -> Database:
    db = Database(problem, database_type="valid")
    db.add_individual(Individual(problem, x=np.zeros(problem.n_input), y=y.copy(), ID=1))
    return db


def plot_decision_and_objective_space(
    base_dir: Path,
    db: Database,
    fronts: list[list[int]],
    ) -> Path:
    xs = db.get_xs()
    ys = get_objectives_matrix(db)
    ranks = collect_ranks(db)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.add_patch(
        plt.Circle((0.0, 0.0), np.sqrt(0.64), fill=False, linestyle="--", color="0.3", label="g=0")
    )
    sc = ax.scatter(xs[:, 0], xs[:, 1], c=ranks, cmap="tab10", s=28, alpha=0.85, edgecolors="k", linewidths=0.2)
    plt.colorbar(sc, ax=ax, label="pareto_rank")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision space (colored by Pareto rank)")
    ax.set_aspect("equal")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1]
    viol = np.array(
        [float(indi.sum_violation or 0.0) if indi.valid_evaluation else np.inf for indi in db.individuals]
    )
    feas = viol <= 0.0
    ax.scatter(
        ys[~feas, 0],
        ys[~feas, 1],
        c="0.75",
        s=22,
        marker="x",
        label="infeasible / failed",
    )
    sc2 = ax.scatter(
        ys[feas, 0],
        ys[feas, 1],
        c=ranks[feas],
        cmap="tab10",
        s=32,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.2,
        label="feasible",
    )
    plt.colorbar(sc2, ax=ax, label="pareto_rank")

    if len(fronts) > 0 and len(fronts[0]) > 0:
        f0 = fronts[0]
        y0 = ys[f0, :]
        order = np.argsort(y0[:, 0])
        ax.plot(y0[order, 0], y0[order, 1], "r-", lw=1.2, alpha=0.7, label="first front (piecewise)")

    ax.set_xlabel("y1 = x1^2 + x2^2  (minimize)")
    ax.set_ylabel("y2 = x1 - x2  (maximize)")
    ax.set_title("Objective space")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = base_dir / "ea_decision_objective.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_crowding_and_selection(
    base_dir: Path,
    db: Database,
    fronts: list[list[int]],
    selected_idx: list[int],
    ) -> Path:
    ys = get_objectives_matrix(db)
    cd = collect_crowding(db)
    ranks = collect_ranks(db)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    if len(fronts) > 0 and len(fronts[0]) > 1:
        idx = fronts[0]
        yf = ys[idx, :]
        cdf = np.array([cd[i] for i in idx])
        finite = np.isfinite(cdf)
        sizes = np.full(len(idx), 40.0)
        sizes[finite] = 20.0 + 180.0 * (cdf[finite] / (np.nanmax(cdf[finite]) + 1e-12))
        ax.scatter(
            yf[~finite, 0],
            yf[~finite, 1],
            s=120,
            marker="*",
            c="gold",
            edgecolors="k",
            linewidths=0.4,
            label="crowding = inf (extremes)",
            zorder=3,
        )
        if np.any(finite):
            ax.scatter(
                yf[finite, 0],
                yf[finite, 1],
                s=sizes[finite],
                c="steelblue",
                alpha=0.75,
                edgecolors="k",
                linewidths=0.2,
                label="finite crowding (marker size)",
            )
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title("First front: crowding distance (NSGA-II)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    sel_set = set(selected_idx)
    for i in range(db.size):
        if i in sel_set:
            continue
        ax.scatter(ys[i, 0], ys[i, 1], c="0.85", s=22, marker="o", zorder=1)
    for i in selected_idx:
        ax.scatter(ys[i, 0], ys[i, 1], c="crimson", s=55, marker="o", edgecolors="k", linewidths=0.3, zorder=2)
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title(f"Environmental selection (n={len(selected_idx)})")
    ax.grid(alpha=0.25)

    fig.suptitle(f"assign_crowding_distance + select_parent_indices (ranks 1..{int(ranks.max())})", fontsize=11)
    fig.tight_layout()
    out = base_dir / "ea_crowding_selection.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_pareto_sort_and_shrink(
    base_dir: Path,
    db_before_shrink: Database,
    db_after_shrink: Database,
    ) -> Path:
    ys_b = get_objectives_matrix(db_before_shrink)
    ys_a = get_objectives_matrix(db_after_shrink)
    ranks_b = collect_ranks(db_before_shrink)
    ranks_a = collect_ranks(db_after_shrink)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    order = np.argsort(ranks_b)
    ax.scatter(
        np.arange(db_before_shrink.size),
        ranks_b[order],
        c="tab:blue",
        s=30,
        label="pareto_rank after rank_pareto (sorted order)",
    )
    ax.set_xlabel("sorted index (best -> worst)")
    ax.set_ylabel("pareto_rank")
    ax.set_title("rank_pareto: non-decreasing rank along sorted list")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.scatter(ys_b[:, 0], ys_b[:, 1], c="0.75", s=24, alpha=0.6, label=f"before shrink (n={db_before_shrink.size})")
    ax.scatter(
        ys_a[:, 0],
        ys_a[:, 1],
        c=ranks_a,
        cmap="viridis",
        s=45,
        edgecolors="k",
        linewidths=0.25,
        label=f"after shrink (n={db_after_shrink.size})",
    )
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title("truncation via select_parent_indices after rank_pareto")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = base_dir / "ea_rank_sort_shrink.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings_dominance_based_algorithm.json"
    build_minimal_settings_file(settings_path=settings_path, work_dir=base_dir)

    data_settings = SettingsData(name="demo_data_ea", fname_settings=str(settings_path))
    problem_settings = SettingsProblem(
        name="demo_problem_ea",
        data_settings=data_settings,
        fname_settings=str(settings_path),
    )
    problem = Problem(data_settings=data_settings, problem_settings=problem_settings)
    SettingsOptimization(name="demo_opt_ea", fname_settings=str(settings_path))

    np.random.seed(42)
    xs = np.random.uniform(-1.0, 1.0, size=(100, problem.n_input))
    db = build_candidate_database(problem=problem, xs=xs, id_start=1)
    db.evaluate_individuals(user_func=user_func)

    verify_faster_valid_flag_on_feasible_subset(db)
    demo_pareto_dominance_print(problem)

    db_work = clone_database(db)
    db_work._is_valid_database = False
    fronts = DominanceBasedAlgorithm.non_dominated_ranking(db_work)
    DominanceBasedAlgorithm.assign_crowding_distance(db_work)

    population_size = 40
    selected = DominanceBasedAlgorithm.select_parent_indices(db_work, n_select=population_size)
    if len(selected) != population_size:
        raise AssertionError("select_parent_indices length mismatch")

    fig1 = plot_decision_and_objective_space(base_dir, db_work, fronts)
    fig2 = plot_crowding_and_selection(base_dir, db_work, fronts, selected)

    db_rank = clone_database(db)
    db_rank._is_valid_database = False
    DominanceBasedAlgorithm.rank_pareto(db_rank)
    if not db_rank.sorted:
        raise AssertionError("rank_pareto should mark database sorted")

    ranks_sorted = collect_ranks(db_rank)
    for i in range(1, db_rank.size):
        a = db_rank.individuals[i - 1]
        b = db_rank.individuals[i]
        if a.pareto_rank > b.pareto_rank:
            raise AssertionError("rank_pareto sort: pareto_rank should be non-decreasing")

    target_n = 55
    # Use ranked `db_rank` directly: a full `clone_database` copy does not carry
    # `_index_pareto_fronts`, which `select_parent_indices` requires.
    idx_keep = DominanceBasedAlgorithm.select_parent_indices(db_rank, n_select=target_n)
    db_shrink = db_rank.get_sub_database(index_list=idx_keep, deepcopy=True)
    if db_shrink.size != target_n:
        raise AssertionError(f"truncation: expected size {target_n}, got {db_shrink.size}")

    db_shrink._is_valid_database = False
    DominanceBasedAlgorithm.rank_pareto(db_shrink)

    fig3 = plot_rank_pareto_sort_and_shrink(base_dir, db_rank, db_shrink)

    n_feas = sum(
        1
        for indi in db.individuals
        if indi.valid_evaluation and indi.sum_violation is not None and indi.sum_violation <= 0.0
    )
    print(f"[DATABASE] n={db.size}, feasible={n_feas}")
    print(f"[VERIFY] feasible subset: is_valid_database True/False: OK (if n_feas>=3)")
    print(f"[RANK] max pareto_rank = {int(ranks_sorted.max())}, fronts = {len(fronts)}")
    print(f"[SELECT] first {population_size} indices by rank + crowding: OK")
    print(f"[OUTPUT] {fig1}")
    print(f"[OUTPUT] {fig2}")
    print(f"[OUTPUT] {fig3}")


if __name__ == "__main__":
    main()
