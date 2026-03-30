'''
Example: comparison of single-objective optimization algorithms.

- Create a problem for benchmark functions:
  1) benchmark functions: Rastrigin (in `aeroopt.utils.benchmark`)
  2) n_input = 10, n_output = 1, n_constraint = 1
  3) xi in [-2, 2]
  4) constraint1: x1^2 + x2^2 - 0.64 <= 0.0

- Load algorithm settings (`SettingsNSGAII`, `SettingsDE`, `SettingsNRBO`).

- Create optimization objects:
  1) use those algorithm settings for crossover/mutation
  2) use mp_evaluation for evaluation
  3) population size = 32
  4) max_iterations = 20

- Start optimization:
  1) run separately for each algorithm
  2) plot all the results in a single figure using subplots
  3) the x-axis is the iteration number, the y-axis is the objective function value
  4) mark feasible individuals in green hollow circles
  5) mark infeasible individuals in gray hollow circles
  6) mark the elite (best) individuals of each iteration in red hollow triangles

'''

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroopt.core import MultiProcessEvaluation, Problem, SettingsData, SettingsProblem
from aeroopt.optimization import (
    OptDE, OptNSGAII, OptNRBO,
    SettingsDE, SettingsNRBO, SettingsNSGAII, SettingsOptimization
)
from aeroopt.utils.benchmark import Rastrigin


N_INPUT = 10
POPULATION_SIZE = 32
MAX_ITERATIONS = 20
MASTER_SEED = 42
Y_BOUNDS = (-20.0, 180.0)


def build_settings_file(settings_path: Path, work_dir: Path) -> None:
    settings = {
        "soo_data": {
            "type": "SettingsData",
            "name": "soo_data",
            "name_input": [f"x{i}" for i in range(1, N_INPUT + 1)],
            "input_low": [-2.0] * N_INPUT,
            "input_upp": [2.0] * N_INPUT,
            "input_precision": [0.0] * N_INPUT,
            "name_output": ["y1"],
            "output_low": [Y_BOUNDS[0]],
            "output_upp": [Y_BOUNDS[1]],
            "output_precision": [0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "soo_problem": {
            "type": "SettingsProblem",
            "name": "soo_problem",
            "name_data_settings": "soo_data",
            "output_type": [-1],
            "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"],
        },
        "soo_opt": {
            "type": "SettingsOptimization",
            "name": "soo_opt",
            "resume": False,
            "population_size": POPULATION_SIZE,
            "max_iterations": MAX_ITERATIONS,
            "fname_db_total": "db-total.json",
            "fname_db_elite": "db-elite.json",
            "fname_db_population": "db-population.json",
            "fname_db_resume": "db-resume.json",
            "fname_log": "optimization.log",
            "working_directory": str(work_dir),
            "info_level_on_screen": 1,
            "critical_potential_x": 0.2,
        },
        "soo_nsgaii_alg": {
            "type": "SettingsNSGAII",
            "name": "soo_nsgaii_alg",
            "cross_rate": 0.8,
            "mut_rate": 0.8,
            "pow_sbx": 20.0,
            "pow_poly": 20.0,
            "reserve_ratio": 0.3,
        },
        "soo_de_alg": {
            "type": "SettingsDE",
            "name": "soo_de_alg",
            "scale_factor": 0.3,
            "cross_rate": 0.8,
        },
        "soo_nrbo_alg": {
            "type": "SettingsNRBO",
            "name": "soo_nrbo_alg",
            "deciding_factor": 0.6,
        },
    }
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def _benchmark_user_func(
    x: np.ndarray,
    bench_fn: Callable[[np.ndarray], float],
    **kwargs,
    ) -> Tuple[bool, np.ndarray]:
    y = float(bench_fn(np.asarray(x, dtype=float)))
    return True, np.array([y], dtype=float)


def is_plot_feasible(indi) -> bool:
    if not indi.valid_evaluation or indi.y is None:
        return False
    return float(indi.sum_violation) <= 0.0


def _best_feasible_by_generation(opt) -> tuple:
    gen_to_best = {}
    for indi in opt.db_total.individuals:
        if not is_plot_feasible(indi):
            continue
        g = int(indi.generation)
        y = float(indi.y[0])
        if g not in gen_to_best or y < gen_to_best[g]:
            gen_to_best[g] = y
    if not gen_to_best:
        return np.array([], dtype=int), np.array([], dtype=float)
    gs = np.array(sorted(gen_to_best.keys()), dtype=int)
    ys = np.array([gen_to_best[int(g)] for g in gs], dtype=float)
    return gs, ys


def plot_subplot(ax, opt, title: str) -> None:
    for indi in opt.db_total.individuals:
        if indi.y is None or indi.y.size < 1:
            continue
        g = int(indi.generation)
        y = float(indi.y[0])
        if is_plot_feasible(indi):
            ax.scatter(
                g, y, s=22, facecolors="none", edgecolors="green",
                linewidths=0.6, zorder=2,
            )
        else:
            ax.scatter(
                g, y, s=22, facecolors="none", edgecolors="0.55",
                linewidths=0.5, zorder=1,
            )

    g_best, y_best = _best_feasible_by_generation(opt)
    if g_best.size > 0:
        ax.scatter(
            g_best, y_best, s=55, marker="^", facecolors="none",
            edgecolors="red", linewidths=1.0, zorder=4,
        )
        ax.plot(g_best, y_best, color="red", linewidth=0.8, alpha=0.5, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel("objective value")
    ax.set_xlim(-1, MAX_ITERATIONS+1)
    ax.set_ylim(Y_BOUNDS)
    ax.grid(True, alpha=0.3)


def run_algorithm(
    algorithm: str,
    work_dir: Path,
    n_proc: int,
    ) -> object:
    work_dir.mkdir(parents=True, exist_ok=True)
    settings_path = work_dir / "settings.json"
    build_settings_file(settings_path, work_dir)

    data = SettingsData("soo_data", fname_settings=str(settings_path))
    problem = Problem(
        data,
        SettingsProblem("soo_problem", data, fname_settings=str(settings_path)),
    )
    opt_settings = SettingsOptimization("soo_opt", fname_settings=str(settings_path))
    opt_settings.working_directory = str(work_dir)
    user_func = functools.partial(_benchmark_user_func, bench_fn=Rastrigin)
    mp_eval = MultiProcessEvaluation(
        dim_input=N_INPUT,
        dim_output=1,
        func=None,
        n_process=n_proc,
        information=False,
    )

    np.random.seed(MASTER_SEED)
    if algorithm == "NSGA-II":
        alg_settings = SettingsNSGAII("soo_nsgaii_alg", fname_settings=str(settings_path))
        stream_seed=MASTER_SEED + 10_001
        opt = OptNSGAII(
            problem=problem,
            optimization_settings=opt_settings,
            algorithm_settings=alg_settings,
            user_func=user_func,
            mp_evaluation=mp_eval,
            rng=np.random.default_rng(int(stream_seed))
        )
    elif algorithm == "DE":
        alg_settings = SettingsDE("soo_de_alg", fname_settings=str(settings_path))
        stream_seed=MASTER_SEED + 20_001
        opt = OptDE(
            problem=problem,
            optimization_settings=opt_settings,
            algorithm_settings=alg_settings,
            user_func=user_func,
            mp_evaluation=mp_eval,
            rng=np.random.default_rng(int(stream_seed))
        )
    elif algorithm == "NRBO":
        alg_settings = SettingsNRBO("soo_nrbo_alg", fname_settings=str(settings_path))
        stream_seed=MASTER_SEED + 30_001
        opt = OptNRBO(
            problem=problem,
            optimization_settings=opt_settings,
            algorithm_settings=alg_settings,
            user_func=user_func,
            mp_evaluation=mp_eval,
            rng=np.random.default_rng(int(stream_seed))
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    opt.main()
    return opt


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    run_root = script_dir / "_soo_work"
    run_root.mkdir(parents=True, exist_ok=True)

    n_proc = os.cpu_count()
    if n_proc is None:
        n_proc = 2
    n_proc = max(1, min(n_proc, 8))

    algos = ["NSGA-II", "DE", "NRBO"]
    fig, axes = plt.subplots(1, len(algos), figsize=(12.5, 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, algo in zip(axes, algos):
        print(f"SOO example: running {algo} ...", flush=True)
        opt = run_algorithm(
            algo,
            run_root / algo.replace("-", "").replace(" ", "_"),
            n_proc=n_proc,
        )
        plot_subplot(ax, opt, algo)

    out_png = script_dir / "soo_algorithms_comparison.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()


