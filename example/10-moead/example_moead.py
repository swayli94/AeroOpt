'''
Example: demonstrate MOEA/D (multiobjective evolutionary algorithm based on decomposition).

- Create a problem for benchmark functions:
  1) benchmark functions: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 (in `AeroOpt.utils.benchmark`)
  2) n_input = 3, n_output = 2, n_constraint = 1
  3) xi in [0, 1]
  4) constraint1: x1^2 + x2^2 - 0.64 <= 0.0（与 DE / RVEA 等示例相同）

- Load MOEA/D algorithm settings (`SettingsMOEAD`).

- Create a MOEA/D optimization object `opt_moead`:
  1) use those algorithm settings for SBX/PM and neighborhood parameters
  2) use mp_evaluation for evaluation
  3) population size = ``examples_common.POPULATION_SIZE``（与 DE 等示例相同；须等于 Das–Dennis 权重个数）
  4) max_iterations = 20

- Start optimization:
  1) run separately for each benchmark function
  2) plot all the results in a single figure using subplots
  3) mark feasible individuals of each iteration in different colors of hollow circles
  4) mark infeasible individuals in gray hollow circles
  5) mark the final Pareto front in red hollow triangles

'''

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from examples_common import (
    MAX_ITERATIONS,
    N_INPUT,
    POPULATION_SIZE,
    PLOT_F1_LIM,
    PLOT_F2_LIM_BY_BENCHMARK,
    apply_benchmark_seeds,
    moead_rng_seed,
)

from AeroOpt.core import Problem, MultiProcessEvaluation, SettingsData, SettingsProblem

from AeroOpt.optimization import SettingsOptimization, SettingsMOEAD, OptMOEAD as OptMOEADBase
from AeroOpt.utils import benchmark as bench


class OptMOEAD(OptMOEADBase):
    '''MOEA/D driver with a fixed NumPy ``Generator`` (library default is unseeded).'''

    def __init__(
            self,
            problem: Problem,
            optimization_settings: SettingsOptimization,
            algorithm_settings: SettingsMOEAD,
            user_func: Callable = None,
            mp_evaluation: MultiProcessEvaluation = None,
            moead_stream_seed: int = 0,
            ):
        super().__init__(
            problem=problem,
            optimization_settings=optimization_settings,
            algorithm_settings=algorithm_settings,
            user_func=user_func,
            mp_evaluation=mp_evaluation,
        )
        self._rng = np.random.default_rng(int(moead_stream_seed))


BENCHMARKS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("ZDT1", bench.ZDT1),
    ("ZDT2", bench.ZDT2),
    ("ZDT3", bench.ZDT3),
    ("ZDT4", bench.ZDT4),
    ("ZDT6", bench.ZDT6),
]


def build_settings_file(
    settings_path: Path, work_dir: Path, benchmark_name: str,
    ) -> None:
    name_inputs = [f"x{i}" for i in range(1, N_INPUT + 1)]
    f2_lo, f2_hi = PLOT_F2_LIM_BY_BENCHMARK[benchmark_name]
    settings = {
        "zdt_moead_data": {
            "type": "SettingsData",
            "name": "zdt_moead_data",
            "name_input": name_inputs,
            "input_low": [0.0] * N_INPUT,
            "input_upp": [1.0] * N_INPUT,
            "input_precision": [0.0] * N_INPUT,
            "name_output": ["y1", "y2"],
            "output_low": [PLOT_F1_LIM[0], f2_lo],
            "output_upp": [PLOT_F1_LIM[1], f2_hi],
            "output_precision": [0.0, 0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "zdt_moead_problem": {
            "type": "SettingsProblem",
            "name": "zdt_moead_problem",
            "name_data_settings": "zdt_moead_data",
            "output_type": [-1, -1],
            "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"],
        },
        "zdt_moead_opt": {
            "type": "SettingsOptimization",
            "name": "zdt_moead_opt",
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
        "zdt_moead_alg": {
            "type": "SettingsMOEAD",
            "name": "zdt_moead_alg",
            "cross_rate": 0.9,
            "mut_rate": 0.9,
            "pow_sbx": 20.0,
            "pow_poly": 20.0,
            "n_partitions": None,
            "n_neighbors": 20,
            "prob_neighbor_mating": 0.9,
            "decomposition": "auto",
            "pbi_theta": 5.0,
        },
    }
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def _benchmark_user_func(
    x: np.ndarray,
    bench_fn: Callable[[np.ndarray], np.ndarray],
    **kwargs,
    ) -> Tuple[bool, np.ndarray]:
    y = bench_fn(np.asarray(x, dtype=float))
    return True, np.asarray(y, dtype=float)


def is_plot_feasible(indi) -> bool:
    if not indi.valid_evaluation or indi.y is None:
        return False
    return float(indi.sum_violation) <= 0.0


def plot_subplot(ax, opt: OptMOEAD, title: str, vmax_gen: int,
                show_pareto_label: bool) -> None:
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(vmax_gen, 1))

    for indi in opt.db_total.individuals:
        if indi.y is None or indi.y.size < 2:
            continue
        y1, y2 = float(indi.y[0]), float(indi.y[1])
        g = int(indi.generation)
        if is_plot_feasible(indi):
            color = cmap(norm(g))
            ax.scatter(
                y1,
                y2,
                s=22,
                facecolors="none",
                edgecolors=color,
                linewidths=0.6,
                zorder=2,
            )
        else:
            ax.scatter(
                y1,
                y2,
                s=22,
                facecolors="none",
                edgecolors="0.55",
                linewidths=0.5,
                zorder=1,
            )

    if opt.db_elite.size > 0:
        ys = np.array(
            [[float(i.y[0]), float(i.y[1])] for i in opt.db_elite.individuals if i.y is not None and i.y.size >= 2],
            dtype=float,
        )
        if ys.size > 0:
            ax.scatter(
                ys[:, 0],
                ys[:, 1],
                s=55,
                marker="^",
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                zorder=4,
                label="Pareto front (elite)" if show_pareto_label else None,
            )
            if show_pareto_label:
                ax.legend(loc="best", fontsize=7)

    ax.set_title(title)
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.grid(True, alpha=0.3)


def run_one_benchmark(
    bench_fn: Callable[[np.ndarray], np.ndarray],
    work_dir: Path,
    mp_eval: MultiProcessEvaluation,
    bench_index: int,
    benchmark_name: str,
    ) -> OptMOEAD:
    work_dir.mkdir(parents=True, exist_ok=True)
    settings_path = work_dir / "settings.json"
    build_settings_file(settings_path, work_dir, benchmark_name)

    data = SettingsData("zdt_moead_data", fname_settings=str(settings_path))
    problem = Problem(
        data,
        SettingsProblem("zdt_moead_problem", data, fname_settings=str(settings_path)),
    )
    opt_settings = SettingsOptimization("zdt_moead_opt", fname_settings=str(settings_path))
    opt_settings.working_directory = str(work_dir)

    alg_settings = SettingsMOEAD("zdt_moead_alg", fname_settings=str(settings_path))
    user_func = functools.partial(_benchmark_user_func, bench_fn=bench_fn)

    opt_moead = OptMOEAD(
        problem=problem,
        optimization_settings=opt_settings,
        algorithm_settings=alg_settings,
        user_func=user_func,
        mp_evaluation=mp_eval,
        moead_stream_seed=moead_rng_seed(bench_index),
    )
    apply_benchmark_seeds(bench_index)
    opt_moead.main()
    return opt_moead


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    run_root = script_dir / "_moead_work"
    run_root.mkdir(parents=True, exist_ok=True)

    n_proc = os.cpu_count()
    if n_proc is None:
        n_proc = 2
    n_proc = max(1, min(n_proc, 8))

    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(16, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mp_eval = MultiProcessEvaluation(
        dim_input=N_INPUT,
        dim_output=2,
        func=None,
        n_process=n_proc,
        information=False,
    )

    for i, (ax, (bname, bfn)) in enumerate(zip(axes, BENCHMARKS)):
        print(f"MOEA/D example: running {bname} ...", flush=True)
        subdir = run_root / bname
        opt_moead = run_one_benchmark(bfn, subdir, mp_eval, bench_index=i, benchmark_name=bname)
        plot_subplot(ax, opt_moead, bname, MAX_ITERATIONS, show_pareto_label=True)
        ax.set_xlim(PLOT_F1_LIM)
        ax.set_ylim(PLOT_F2_LIM_BY_BENCHMARK[bname])

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=MAX_ITERATIONS)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85, aspect=30, pad=0.02)
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    cbar.set_label("generation")

    out_png = script_dir / "moead_zdt_subplots.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
