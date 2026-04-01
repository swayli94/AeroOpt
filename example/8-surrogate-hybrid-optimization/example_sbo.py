'''
Example: demonstrate the Surrogate-based Optimization (SBO) algorithm.

- Create a problem for benchmark functions:
  1) benchmark functions: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 (in `aeroopt.utils.benchmark`)
  2) n_input = 3, n_output = 2, n_constraint = 1
  3) xi in [0, 1]
  4) constraint1: x1^2 + x2^2 - 0.64 <= 0.0

- Use `aeroopt.optimization.hybrid.sbo.SBO` as the main optimization framework.
  1) population size = 32
  2) max_iterations = 20
  3) use mp_evaluation for evaluation

- Use `aeroopt.utils.surrogate.Kriging` as the surrogate model.
  1) train on the scaled input/output data
  2) use the default parameters of the `KPLS` model in `SMT` package
  3) use the same problem as the global optimization problem

- Use `aeroopt.optimization.stochastic.de.OptDE` as the optimization algorithm on the surrogate model.
  1) population size = 64
  2) max_iterations = 20

- Use `aeroopt.optimization.hybrid.sbo.PostProcessSBO` to evaluate the performance of the surrogate model.

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
    MASTER_SEED,
    MAX_ITERATIONS,
    N_INPUT,
    POPULATION_SIZE,
    PLOT_F1_LIM,
    PLOT_F2_LIM_BY_BENCHMARK,
    apply_benchmark_seeds,
)

from aeroopt.core import Problem, MultiProcessEvaluation, SettingsData, SettingsProblem

from aeroopt.optimization import SettingsOptimization, SettingsDE
from aeroopt.optimization.hybrid.sbo import PostProcessSBO, SBO
from aeroopt.optimization.stochastic.de import OptDE
from aeroopt.utils import benchmark as bench
from aeroopt.utils.surrogate import Kriging


SBO_INNER_POPULATION_SIZE = 64
SBO_INNER_MAX_ITERATIONS = 10


def sbo_inner_de_rng_seed(bench_index: int) -> int:
    return MASTER_SEED + bench_index * 1_000_000 + 70_000_001


BENCHMARKS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("ZDT1", bench.ZDT1),
    ("ZDT2", bench.ZDT2),
    ("ZDT3", bench.ZDT3),
    ("ZDT4", bench.ZDT4),
    ("ZDT6", bench.ZDT6),
]


def build_settings_file(
    settings_path: Path, work_dir: Path, inner_work_dir: Path, benchmark_name: str,
    ) -> None:
    name_inputs = [f"x{i}" for i in range(1, N_INPUT + 1)]
    f2_lo, f2_hi = PLOT_F2_LIM_BY_BENCHMARK[benchmark_name]
    settings = {
        "zdt_sbo_data": {
            "type": "SettingsData",
            "name": "zdt_sbo_data",
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
        "zdt_sbo_problem": {
            "type": "SettingsProblem",
            "name": "zdt_sbo_problem",
            "name_data_settings": "zdt_sbo_data",
            "output_type": [-1, -1],
            "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"],
        },
        "zdt_sbo_opt": {
            "type": "SettingsOptimization",
            "name": "zdt_sbo_opt",
            "resume": False,
            "population_size": POPULATION_SIZE,
            "max_iterations": MAX_ITERATIONS,
            "fname_db_total": "db-total.json",
            "fname_db_elite": "db-elite.json",
            "fname_db_population": "db-population.json",
            "fname_db_resume": "db-resume.json",
            "fname_log": "optimization.log",
            "working_directory": str(work_dir),
            "info_level_on_screen": 2,
            "critical_potential_x": 0.2,
        },
        "zdt_sbo_inner_opt": {
            "type": "SettingsOptimization",
            "name": "zdt_sbo_inner_opt",
            "resume": False,
            "population_size": SBO_INNER_POPULATION_SIZE,
            "max_iterations": SBO_INNER_MAX_ITERATIONS,
            "fname_db_total": "db-total-inner.json",
            "fname_db_elite": "db-elite-inner.json",
            "fname_db_population": "db-population-inner.json",
            "fname_db_resume": "db-resume-inner.json",
            "fname_log": "optimization-inner.log",
            "working_directory": str(inner_work_dir),
            "info_level_on_screen": 0,
            "critical_potential_x": 0.2,
        },
        "zdt_sbo_alg": {
            "type": "SettingsDE",
            "name": "zdt_sbo_alg",
            "scale_factor": 0.5,
            "cross_rate": 0.9,
        },
    }
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def _benchmark_user_func_batch(
    xs: np.ndarray,
    bench_fn: Callable[[np.ndarray], np.ndarray],
    **kwargs,
    ) -> Tuple[list, np.ndarray]:
    '''
    Batch evaluation in the main process (used with ``user_func_supports_parallel``),
    avoiding Windows process-pool pickling issues with ``functools.partial``.
    '''
    xs = np.asarray(xs, dtype=float)
    n = xs.shape[0]
    ys = np.zeros((n, 2), dtype=float)
    for i in range(n):
        ys[i, :] = bench_fn(xs[i, :])
    return [True] * n, ys


def is_plot_feasible(indi) -> bool:
    if not indi.valid_evaluation or indi.y is None:
        return False
    return float(indi.sum_violation) <= 0.0


def plot_subplot(ax, opt: SBO, title: str, vmax_gen: int,
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
    ) -> SBO:
    work_dir.mkdir(parents=True, exist_ok=True)
    inner_work_dir = work_dir / "inner_de"
    inner_work_dir.mkdir(parents=True, exist_ok=True)

    settings_path = work_dir / "settings.json"
    build_settings_file(settings_path, work_dir, inner_work_dir, benchmark_name)

    data = SettingsData("zdt_sbo_data", fname_settings=str(settings_path))
    problem = Problem(
        data,
        SettingsProblem("zdt_sbo_problem", data, fname_settings=str(settings_path)),
    )
    opt_settings = SettingsOptimization("zdt_sbo_opt", fname_settings=str(settings_path))
    opt_settings.working_directory = str(work_dir)

    inner_opt_settings = SettingsOptimization("zdt_sbo_inner_opt", fname_settings=str(settings_path))
    inner_opt_settings.working_directory = str(inner_work_dir)

    alg_settings = SettingsDE("zdt_sbo_alg", fname_settings=str(settings_path))
    user_func = functools.partial(_benchmark_user_func_batch, bench_fn=bench_fn)

    surrogate = Kriging(problem, train_on_scaled_data=True)

    opt_inner = OptDE(
        problem=problem,
        optimization_settings=inner_opt_settings,
        algorithm_settings=alg_settings,
        user_func=None,
        user_func_supports_parallel=True,
        mp_evaluation=None,
        rng=np.random.default_rng(int(sbo_inner_de_rng_seed(bench_index))),
        save_result_files=False,
        logging=False,
    )

    sbo = SBO(
        problem=problem,
        optimization_settings=opt_settings,
        surrogate=surrogate,
        opt_on_surrogate=opt_inner,
        user_func=user_func,
        mp_evaluation=mp_eval
    )
    sbo.user_func_supports_parallel = True
    sbo.post_process = PostProcessSBO(sbo, surrogate)

    apply_benchmark_seeds(bench_index)
    sbo.main()
    return sbo


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    run_root = script_dir / "_sbo_work"
    run_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(16, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # Serial pool (``n_process=None``): real evaluations use the batch user_func in-process.
    # Process pools on Windows often break with pickled ``partial`` user functions.
    mp_eval = MultiProcessEvaluation(
        dim_input=N_INPUT,
        dim_output=2,
        func=None,
        n_process=None,
        information=False,
    )

    for i, (ax, (bname, bfn)) in enumerate(zip(axes, BENCHMARKS)):
        print(f"SBO example: running {bname} ...", flush=True)
        subdir = run_root / bname
        opt_sbo = run_one_benchmark(bfn, subdir, mp_eval, bench_index=i, benchmark_name=bname)
        plot_subplot(ax, opt_sbo, bname, MAX_ITERATIONS, show_pareto_label=True)
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

    out_png = script_dir / "sbo_zdt_subplots.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
