'''
Example: demonstrate multi-process evaluation of a user-defined function.

- Define a built-in (python) user-defined function.
- Define an external user-defined function (both Linux and Windows are supported).
- Create a multi-process evaluation object.
- Create a numpy array of input points (randomly generated).
- Evaluate the user-defined function using the multi-process evaluation object.
- Run these points using both built-in and external user-defined functions, respectively. (Run two rounds in total)
- Save the results in two databases and write to two JSON/Excel files, respectively.
'''

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroopt.core.database import Database
from aeroopt.core.individual import Individual
from aeroopt.core.mpEvaluation import MultiProcessEvaluation
from aeroopt.core.problem import Problem
from aeroopt.core.settings import SettingsData, SettingsProblem


def build_minimal_settings_file(settings_path: Path) -> None:
    settings = {
        "data_demo_mp": {
            "type": "SettingsData",
            "name": "demo_data_mp",
            "name_input": ["x1", "x2"],
            "input_low": [0.0, 0.0],
            "input_upp": [1.0, 1.0],
            "input_precision": [0.0, 0.0],
            "name_output": ["y1"],
            "output_low": [0.0],
            "output_upp": [10.0],
            "output_precision": [0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "problem_demo_mp": {
            "type": "SettingsProblem",
            "name": "demo_problem_mp",
            "name_data_settings": "demo_data_mp",
            "output_type": ["-1"],
            "constraint_strings": [],
        },
    }

    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def builtin_func(x: np.ndarray, **kwargs):
    # y1 = x1^2 + x2^2
    return True, np.array([float(np.sum(x**2))], dtype=float)


def prepare_external_runfiles(base_dir: Path) -> None:
    runfiles_dir = base_dir / "Runfiles"
    runfiles_dir.mkdir(parents=True, exist_ok=True)

    external_py = runfiles_dir / "external_evaluator.py"
    external_py.write_text(
        (
            "from pathlib import Path\n"
            "\n"
            "def read_input(input_path: Path):\n"
            "    vals = {}\n"
            "    for line in input_path.read_text(encoding='utf-8').splitlines():\n"
            "        parts = line.split()\n"
            "        if len(parts) >= 2:\n"
            "            vals[parts[0]] = float(parts[1])\n"
            "    return vals\n"
            "\n"
            "def main():\n"
            "    cwd = Path.cwd()\n"
            "    data = read_input(cwd / 'input.txt')\n"
            "    x1 = data.get('x1', 0.0)\n"
            "    x2 = data.get('x2', 0.0)\n"
            "    y1 = x1 * x1 + x2 * x2\n"
            "    with (cwd / 'output.txt').open('w', encoding='utf-8') as f:\n"
            "        f.write(f'y1 {y1}\\n')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        encoding="utf-8",
    )

    run_bat = runfiles_dir / "run.bat"
    run_bat.write_text(
        "@echo off\r\n"
        "python external_evaluator.py\r\n",
        encoding="utf-8",
    )

    run_sh = runfiles_dir / "run.sh"
    run_sh.write_text(
        "#!/usr/bin/env sh\n"
        "python external_evaluator.py\n",
        encoding="utf-8",
    )
    # Linux/macOS external run expects a runnable run.sh in each case folder.
    run_sh.chmod(0o755)


def prepare_calculation_folders(base_dir: Path, list_name: list[str], is_windows: bool) -> None:
    calculation_dir = base_dir / "Calculation"
    calculation_dir.mkdir(parents=True, exist_ok=True)
    for name in list_name:
        case_dir = calculation_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_dir / "Runfiles" / "external_evaluator.py", case_dir / "external_evaluator.py")
        if is_windows:
            shutil.copy2(base_dir / "Runfiles" / "run.bat", case_dir / "run.bat")
        else:
            run_sh_case = case_dir / "run.sh"
            shutil.copy2(base_dir / "Runfiles" / "run.sh", run_sh_case)
            run_sh_case.chmod(0o755)


def build_database(problem: Problem, xs: np.ndarray, ys: np.ndarray, valid_flags: list[bool]) -> Database:
    db = Database(problem=problem, database_type="total")
    individuals = []
    for idx in range(xs.shape[0]):
        indi = Individual(problem=problem, x=xs[idx, :], y=ys[idx, :], ID=idx + 1)
        indi.valid_evaluation = bool(valid_flags[idx])
        indi.eval_constraints()
        individuals.append(indi)
    db.individuals = individuals
    db.update_id_list()
    return db


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings_mp_example.json"

    db_builtin_json = base_dir / "database_mp_builtin.json"
    db_builtin_excel = base_dir / "database_mp_builtin.xlsx"
    db_external_json = base_dir / "database_mp_external.json"
    db_external_excel = base_dir / "database_mp_external.xlsx"

    # Ensure relative paths in Problem.external_run are anchored here.
    # It expects ./Runfiles and ./Calculation under current working directory.
    original_cwd = Path.cwd()
    try:
        os_name = platform.system().lower()
        is_windows = os_name.startswith("win")
        print(f"[SYSTEM] detected OS: {platform.system()}")

        build_minimal_settings_file(settings_path)
        data_settings = SettingsData(name="demo_data_mp", fname_settings=str(settings_path))
        problem_settings = SettingsProblem(
            name="demo_problem_mp",
            data_settings=data_settings,
            fname_settings=str(settings_path),
        )
        problem = Problem(data_settings=data_settings, problem_settings=problem_settings)

        np.random.seed(42)
        xs = np.random.rand(12, problem.n_input)

        # 1) Built-in function evaluation (multi-process)
        mp_builtin = MultiProcessEvaluation(
            dim_input=problem.n_input,
            dim_output=problem.n_output,
            func=builtin_func,
            n_process=4,
            information=True,
        )
        succeed_builtin, ys_builtin = mp_builtin.evaluate(xs)

        db_builtin = build_database(problem=problem, xs=xs, ys=ys_builtin, valid_flags=succeed_builtin)
        db_builtin.output_database_json(str(db_builtin_json))
        db_builtin.json_to_excel(str(db_builtin_json), str(db_builtin_excel))
        print(f"[BUILTIN] succeed: {sum(succeed_builtin)}/{len(succeed_builtin)}")
        print(f"[BUILTIN] json: {db_builtin_json}")
        print(f"[BUILTIN] excel: {db_builtin_excel}")

        # 2) External function evaluation (multi-process)
        os.chdir(base_dir)
        shutil.rmtree(base_dir / "Calculation", ignore_errors=True)
        prepare_external_runfiles(base_dir)

        list_name = [f"case_{i + 1:03d}" for i in range(xs.shape[0])]
        prepare_calculation_folders(base_dir, list_name, is_windows=is_windows)
        mp_external = MultiProcessEvaluation(
            dim_input=problem.n_input,
            dim_output=problem.n_output,
            func=None,
            n_process=4,
            information=True,
        )
        succeed_external, ys_external = mp_external.evaluate(xs, list_name=list_name, prob=problem)

        db_external = build_database(problem=problem, xs=xs, ys=ys_external, valid_flags=succeed_external)
        db_external.output_database_json(str(db_external_json))
        db_external.json_to_excel(str(db_external_json), str(db_external_excel))
        print(f"[EXTERNAL] succeed: {sum(succeed_external)}/{len(succeed_external)}")
        print(f"[EXTERNAL] json: {db_external_json}")
        print(f"[EXTERNAL] excel: {db_external_excel}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
