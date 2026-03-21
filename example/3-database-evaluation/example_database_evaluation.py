"""
Example: demonstrate evaluation of a database.

- Create a problem with one objective and one constraint.
- Randomly generate input points and build a database from them.
- Use `Database.evaluate_individuals` for:
  1) serial built-in function evaluation
  2) multi-process built-in function evaluation
  3) multi-process external script evaluation
- Save all results to JSON/Excel files.
"""

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

from AeroOpt.core.database import Database
from AeroOpt.core.individual import Individual
from AeroOpt.core.mpEvaluation import MultiProcessEvaluation
from AeroOpt.core.problem import Problem
from AeroOpt.core.settings import SettingsData, SettingsProblem


def build_minimal_settings_file(settings_path: Path) -> None:
    settings = {
        "data_demo_db_eval": {
            "type": "SettingsData",
            "name": "demo_data_db_eval",
            "name_input": ["x1", "x2"],
            "input_low": [0.0, 0.0],
            "input_upp": [1.0, 1.0],
            "input_precision": [0.0, 0.0],
            "name_output": ["y1"],
            "output_low": [0.0],
            "output_upp": [2.0],
            "output_precision": [0.0],
            "critical_scaled_distance": 1.0e-8,
        },
        "problem_demo_db_eval": {
            "type": "SettingsProblem",
            "name": "demo_problem_db_eval",
            "name_data_settings": "demo_data_db_eval",
            "output_type": ["-1"],  # minimize y1
            # Constraint string parser expects tokenized expression with spaces.
            # Meaning: x1 + x2 <= 1.2
            "constraint_strings": ["x1 + x2 - 1.2"],
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


def build_input_database(problem: Problem, xs: np.ndarray) -> Database:
    db = Database(problem=problem, database_type="total")
    db.individuals = [
        Individual(problem=problem, x=xs[i, :], ID=i + 1)
        for i in range(xs.shape[0])
    ]
    db.update_id_list()
    return db


def clone_input_database(problem: Problem, xs: np.ndarray) -> Database:
    # Build independent database objects so each evaluation mode is clear.
    return build_input_database(problem=problem, xs=xs.copy())


def export_and_print(db: Database, json_path: Path, excel_path: Path, tag: str) -> None:
    db.output_database_json(str(json_path))
    db.json_to_excel(str(json_path), str(excel_path))
    n_ok = sum(1 for indi in db.individuals if indi.valid_evaluation)
    print(f"[{tag}] succeed: {n_ok}/{db.size}")
    print(f"[{tag}] json: {json_path}")
    print(f"[{tag}] excel: {excel_path}")


def compare_database_json(file_a: Path, file_b: Path, tag: str, atol: float = 1.0e-9) -> bool:
    with file_a.open("r", encoding="utf-8") as fa:
        data_a = json.load(fa)
    with file_b.open("r", encoding="utf-8") as fb:
        data_b = json.load(fb)

    individuals_a = data_a.get("individuals", [])
    individuals_b = data_b.get("individuals", [])
    if len(individuals_a) != len(individuals_b):
        print(f"[COMPARE:{tag}] size mismatch: {len(individuals_a)} vs {len(individuals_b)}")
        return False

    for idx, (ia, ib) in enumerate(zip(individuals_a, individuals_b), start=1):
        if ia.get("ID") != ib.get("ID"):
            print(f"[COMPARE:{tag}] ID mismatch at index {idx}: {ia.get('ID')} vs {ib.get('ID')}")
            return False
        if ia.get("valid_evaluation") != ib.get("valid_evaluation"):
            print(f"[COMPARE:{tag}] valid_evaluation mismatch at ID={ia.get('ID')}")
            return False

        ya = ia.get("y")
        yb = ib.get("y")
        if ya is None or yb is None:
            if ya != yb:
                print(f"[COMPARE:{tag}] y mismatch at ID={ia.get('ID')}: {ya} vs {yb}")
                return False
        else:
            if not np.allclose(np.asarray(ya, dtype=float), np.asarray(yb, dtype=float), atol=atol, rtol=0.0):
                print(f"[COMPARE:{tag}] y mismatch at ID={ia.get('ID')}: {ya} vs {yb}")
                return False

    print(f"[COMPARE:{tag}] PASS")
    return True


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings_database_evaluation_example.json"
    original_cwd = Path.cwd()

    try:
        os_name = platform.system().lower()
        is_windows = os_name.startswith("win")
        print(f"[SYSTEM] detected OS: {platform.system()}")

        build_minimal_settings_file(settings_path)
        data_settings = SettingsData(name="demo_data_db_eval", fname_settings=str(settings_path))
        problem_settings = SettingsProblem(
            name="demo_problem_db_eval",
            data_settings=data_settings,
            fname_settings=str(settings_path),
        )
        problem = Problem(data_settings=data_settings, problem_settings=problem_settings)

        np.random.seed(2026)
        xs = np.random.rand(12, problem.n_input)

        # 1) Serial built-in evaluation
        db_serial_builtin = clone_input_database(problem=problem, xs=xs)
        db_serial_builtin.evaluate_individuals(user_func=builtin_func)
        serial_json = base_dir / "database_eval_serial_builtin.json"
        serial_excel = base_dir / "database_eval_serial_builtin.xlsx"
        export_and_print(
            db_serial_builtin,
            serial_json,
            serial_excel,
            "SERIAL-BUILTIN",
        )

        # 2) Multi-process built-in evaluation
        db_mp_builtin = clone_input_database(problem=problem, xs=xs)
        mp_builtin = MultiProcessEvaluation(
            dim_input=problem.n_input,
            dim_output=problem.n_output,
            func=builtin_func,
            n_process=4,
            information=True,
        )
        db_mp_builtin.evaluate_individuals(mp_evaluation=mp_builtin, user_func=builtin_func)
        mp_builtin_json = base_dir / "database_eval_mp_builtin.json"
        mp_builtin_excel = base_dir / "database_eval_mp_builtin.xlsx"
        export_and_print(
            db_mp_builtin,
            mp_builtin_json,
            mp_builtin_excel,
            "MP-BUILTIN",
        )

        # 3) Multi-process external evaluation
        os.chdir(base_dir)
        shutil.rmtree(base_dir / "Calculation", ignore_errors=True)
        prepare_external_runfiles(base_dir)

        db_mp_external = clone_input_database(problem=problem, xs=xs)
        list_name = [f"db_case_{indi.ID:03d}" for indi in db_mp_external.individuals]
        prepare_calculation_folders(base_dir, list_name, is_windows=is_windows)
        mp_external = MultiProcessEvaluation(
            dim_input=problem.n_input,
            dim_output=problem.n_output,
            func=None,
            n_process=4,
            information=True,
        )
        db_mp_external.evaluate_individuals(
            mp_evaluation=mp_external,
            user_func=None,
            prefix_folder_name="db_case_",
        )
        mp_external_json = base_dir / "database_eval_mp_external.json"
        mp_external_excel = base_dir / "database_eval_mp_external.xlsx"
        export_and_print(
            db_mp_external,
            mp_external_json,
            mp_external_excel,
            "MP-EXTERNAL",
        )

        same_serial_vs_mp_builtin = compare_database_json(
            serial_json, mp_builtin_json, "SERIAL-BUILTIN vs MP-BUILTIN"
        )
        same_serial_vs_mp_external = compare_database_json(
            serial_json, mp_external_json, "SERIAL-BUILTIN vs MP-EXTERNAL"
        )
        print(
            "[COMPARE] overall:",
            "PASS" if (same_serial_vs_mp_builtin and same_serial_vs_mp_external) else "FAIL",
        )

        print("[DONE] example_database_evaluation finished.")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
