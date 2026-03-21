"""
Example: demonstrate pre-processing of a database `db_candidate`.

- Create a problem `problem`:
  1) n_input=2, n_output=1, n_constraint=2
  2) x1, x2 in [-1, 1]
  3) y1 = x1^2 + x2^2 (use built-in function `user_func`)
  4) constraint1: x1^2 + x2^2 - 0.64 <= 0.0
  5) constraint2: 0.09 - y1 <= 0.0

- Create a OptBaseFramework object `opt`:
  1) use `problem`
  2) use built-in function `user_func` for evaluation
  3) use MultiProcessEvaluation with 4 cores for parallel evaluation

- Define a custom PreProcess class `CustomPreProcess`:
  1) define the pre-processing method `apply`
  2) first `_check_pre_processing_feasibility` in `apply`
  3) second `_adjust_x_values_by_valid_database` in `apply`
  4) then update the `opt.db_candidate` database in place

- Create a CustomPreProcess object `custom_pre_process`:
  1) use `opt`
  2) assign to `opt.pre_process`

- Create a big database `opt.db_total`:
  1) generate 100 random input points `initial_xs`
  2) use `initial_xs` to build `opt.db_candidate`
  3) evaluate candidates with `opt.evaluate_db_candidate`
  4) update with `opt.update_total_and_valid_with_candidate`

- Create a small database `opt.db_candidate` for testing:
  1) generate 10 random input points `xs`
  2) use `xs` to build `opt.db_candidate`
  3) save `xs` for plotting

- Test the pre-processing:
  1) apply the pre-processing to `opt.db_candidate` by `custom_pre_process.apply`
  2) evaluate the `opt.db_candidate` by `opt.evaluate_db_candidate`
  3) save the results in a database and write to a JSON/Excel file

- Plot the results:
  1) create a 2d contour plot of the function `user_func`
  2) plot the constraints as dashed lines
  3) plot the `opt.db_total` as hollow grey circles
  4) plot the `opt.db_valid` as solid black circles
  5) plot the `xs` as hollow blue triangles
  6) plot the `opt.db_candidate` as solid red triangles
  7) plot legend and axis labels
  8) save the plot as a PNG file
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AeroOpt.core.database import Database
from AeroOpt.core.individual import Individual
from AeroOpt.core.mpEvaluation import MultiProcessEvaluation
from AeroOpt.core.problem import Problem
from AeroOpt.core.settings import CustomConstraintFunction, SettingsData, SettingsOptimization, SettingsProblem
from AeroOpt.optimization.base import OptBaseFramework, PreProcess


def user_func(x: np.ndarray, **kwargs):
    # y1 = x1^2 + x2^2
    return True, np.array([np.sum(x**2)], dtype=float)


class DemoOpt(OptBaseFramework):
    """Minimal concrete subclass only for this example."""

    def initialize_population(self) -> None:
        self.iteration = 1

    def generate_candidate_individuals(self) -> None:
        return None

    def select_elite_from_valid(self) -> None:
        self.db_elite = Database(self.problem, database_type="elite")


class CustomPreProcess(PreProcess):
    """Pre-process candidate x values with feasibility and adjustment checks."""

    def apply(self) -> None:
                
        xs = self.opt.db_candidate.get_xs()
        if xs is None or xs.shape[0] == 0:
            return None
        
        self.opt.log(f'Pre-processing of {xs.shape[0]} candidates started.', level=1)

        feasibility_flags, ID_list = self._check_pre_processing_feasibility(
            xs=xs,
            pre_processing_problem=self.opt.problem,
            user_pre_processing_func=self.opt.user_func,
        )
        xs_new = self._adjust_x_values_by_valid_database(
            xs=xs,
            feasibility_flags=feasibility_flags,
            min_scaled_distance=0.01,
            max_scaled_distance=0.05,
            ID_list=ID_list
        )

        for i, indi in enumerate(self.opt.db_candidate.individuals):
            indi.x = xs_new[i, :].copy()
            indi.valid_evaluation = False
            indi.y = None
            indi.sum_violation = None

        return None


def build_minimal_settings_file(settings_path: Path, work_dir: Path) -> None:
    settings = {
        "data_demo_pre": {
            "type": "SettingsData",
            "name": "demo_data_pre",
            "name_input": ["x1", "x2"],
            "input_low": [-1.0, -1.0],
            "input_upp": [1.0, 1.0],
            "input_precision": [0.0, 0.0],
            "name_output": ["y1"],
            "output_low": [0.0],
            "output_upp": [2.0],
            "output_precision": [0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "problem_demo_pre": {
            "type": "SettingsProblem",
            "name": "demo_problem_pre",
            "name_data_settings": "demo_data_pre",
            "output_type": ["-1"],
            "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64", "0.09 - y1"],
        },
        "opt_demo_pre": {
            "type": "SettingsOptimization",
            "name": "demo_opt_pre",
            "resume": False,
            "population_size": 10,
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


def plot_results(base_dir: Path, db_total: Database, db_valid: Database,
        xs_before: np.ndarray, xs_after: np.ndarray,) -> Path:
    
    fig, ax = plt.subplots(figsize=(8, 7))

    grid = np.linspace(-1.0, 1.0, 201)
    xx, yy = np.meshgrid(grid, grid)
    zz = xx**2 + yy**2
    contour = ax.contourf(xx, yy, zz, levels=25, cmap="viridis", alpha=0.85)
    fig.colorbar(contour, ax=ax, label="y1 = x1^2 + x2^2")

    # constraint1: x1^2 + x2^2 = 0.64
    c1 = plt.Circle((0.0, 0.0), np.sqrt(0.64), fill=False, linestyle="--", linewidth=1.5, color="white")
    ax.add_patch(c1)
    # constraint2: y1 = 0.09 => x1^2 + x2^2 = 0.09
    c2 = plt.Circle((0.0, 0.0), np.sqrt(0.09), fill=False, linestyle="--", linewidth=1.5, color="white")
    ax.add_patch(c2)

    if db_total.size > 0:
        x_total = db_total.get_xs()
        ax.scatter(
            x_total[:, 0],
            x_total[:, 1],
            s=36,
            marker="o",
            facecolors="none",
            edgecolors="grey",
            linewidths=1.0,
            label="db_total",
        )
    if db_valid.size > 0:
        x_valid = db_valid.get_xs()
        ax.scatter(
            x_valid[:, 0],
            x_valid[:, 1],
            s=16,
            marker="o",
            c="black",
            linewidths=0.0,
            label="db_valid",
        )

    ax.scatter(
        xs_before[:, 0],
        xs_before[:, 1],
        s=72,
        marker="^",
        facecolors="none",
        edgecolors="dodgerblue",
        linewidths=1.4,
        label="candidate_before",
    )
    ax.scatter(
        xs_after[:, 0],
        xs_after[:, 1],
        s=32,
        marker="^",
        c="red",
        linewidths=0.0,
        label="candidate_after",
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Pre-process demonstration")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    fig_path = base_dir / "pre_process_demo.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings_pre_process_example.json"
    db_json = base_dir / "database_pre_process.json"
    db_excel = base_dir / "database_pre_process.xlsx"

    build_minimal_settings_file(settings_path=settings_path, work_dir=base_dir)

    data_settings = SettingsData(name="demo_data_pre", fname_settings=str(settings_path))
    problem_settings = SettingsProblem(
        name="demo_problem_pre",
        data_settings=data_settings,
        fname_settings=str(settings_path),
    )
    problem = Problem(data_settings=data_settings, problem_settings=problem_settings)
    optimization_settings = SettingsOptimization(name="demo_opt_pre", fname_settings=str(settings_path))

    mp_eval = MultiProcessEvaluation(
        dim_input=problem.n_input,
        dim_output=problem.n_output,
        func=user_func,
        n_process=4,
        information=True,
    )
    opt = DemoOpt(
        problem=problem,
        optimization_settings=optimization_settings,
        user_func=user_func,
        mp_evaluation=mp_eval,
    )
    custom_pre_process = CustomPreProcess(opt=opt)
    opt.pre_process = custom_pre_process

    # Build reference database for adjustment target (db_total / db_valid).
    np.random.seed(2026)
    initial_xs = np.random.uniform(-1.0, 1.0, size=(100, problem.n_input))
    opt.db_candidate = build_candidate_database(problem=problem, xs=initial_xs, id_start=1)
    opt.evaluate_db_candidate()
    opt.update_total_and_valid_with_candidate()

    # Build small testing candidate database.
    xs = np.random.uniform(-1.0, 1.0, size=(10, problem.n_input))
    opt.db_candidate = build_candidate_database(problem=problem, xs=xs, id_start=1001)
    xs_before = xs.copy()

    custom_pre_process.apply()
    xs_after = opt.db_candidate.get_xs().copy()
    opt.evaluate_db_candidate()

    # Save and export post-processed candidates.
    opt.db_candidate.output_database_json(str(db_json))
    opt.db_candidate.json_to_excel(str(db_json), str(db_excel))

    fig_path = plot_results(
        base_dir=base_dir,
        db_total=opt.db_total,
        db_valid=opt.db_valid,
        xs_before=xs_before,
        xs_after=xs_after,
    )

    n_feasible = sum(
        bool(indi.valid_evaluation) and float(indi.sum_violation or 0.0) <= 0.0
        for indi in opt.db_candidate.individuals
    )
    print(f"[CANDIDATE] size: {opt.db_candidate.size}")
    print(f"[CANDIDATE] feasible: {n_feasible}/{opt.db_candidate.size}")
    print(f"[OUTPUT] json: {db_json}")
    print(f"[OUTPUT] excel: {db_excel}")
    print(f"[OUTPUT] figure: {fig_path}")


if __name__ == "__main__":
    main()
