'''
Example: demonstrate core functions of AeroOpt.

- Create a problem and a database.
- Write the database to a JSON file.
- Read the database from a JSON file.
- Convert the JSON database file to an Excel file.
'''

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AeroOpt.core.database import Database
from AeroOpt.core.individual import Individual
from AeroOpt.core.problem import Problem
from AeroOpt.core.settings import CustomConstraintFunction, SettingsData, SettingsProblem


class DemoConstraintYUpper(CustomConstraintFunction):
    """Custom constraint: y1 <= 40."""
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(y[0] - 40.0)


def build_minimal_settings_file(settings_path: Path) -> None:
    """Create a minimal settings file required by SettingsData/SettingsProblem."""
    settings = {
        "data_demo": {
            "type": "SettingsData",
            "name": "demo_data",
            "name_input": ["x1", "x2"],
            "input_low": [0.0, 0.0],
            "input_upp": [10.0, 10.0],
            "input_precision": [0.0, 0.0],
            "name_output": ["y1"],
            "output_low": [0.0],
            "output_upp": [100.0],
            "output_precision": [0.0],
            "critical_scaled_distance": 1.0e-6,
        },
        "problem_demo": {
            "type": "SettingsProblem",
            "name": "demo_problem",
            "name_data_settings": "demo_data",
            "output_type": ["-1"],  # minimize y1
            # String constraints must be tokenized by spaces for this parser.
            # Meaning: x1 + x2 <= 8.0
            "constraint_strings": ["x1 + x2 - 8.0"],
        },
    }

    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings_example.json"
    db_json_path = base_dir / "database_example.json"
    db_excel_path = base_dir / "database_example.xlsx"

    build_minimal_settings_file(settings_path)

    # 1) Define problem and database
    data_settings = SettingsData(name="demo_data", fname_settings=str(settings_path))
    problem_settings = SettingsProblem(
        name="demo_problem",
        data_settings=data_settings,
        fname_settings=str(settings_path),
    )
    # Add custom constraint function: y1 <= 40
    problem_settings.constraint_functions.append(DemoConstraintYUpper(data_settings))
    problem = Problem(data_settings=data_settings, problem_settings=problem_settings)
    db = Database(problem=problem, database_type="total")

    # 2) Add a few individuals
    samples = [
        (1, np.array([1.0, 2.0]), np.array([5.0])),
        (2, np.array([3.0, 4.0]), np.array([25.0])),
        (3, np.array([7.5, 1.5]), np.array([58.5])),
    ]
    db.individuals = [
        Individual(problem=problem, x=x, y=y, ID=indi_id)
        for indi_id, x, y in samples
    ]
    for indi in db.individuals:
        indi.eval_constraints()
    db.update_id_list()

    # 3) Write database to JSON
    db.output_database_json(str(db_json_path))
    print(f"[WRITE] database saved to: {db_json_path}")

    # 4) Read database from JSON into a new database object
    db_loaded = Database(problem=problem, database_type="default")
    db_loaded.read_database_json(str(db_json_path))
    print(f"[READ] loaded size: {db_loaded.size}")
    print(f"[READ] loaded IDs: {[indi.ID for indi in db_loaded.individuals]}")
    print(f"[READ] loaded xs:\n{db_loaded.get_xs()}")
    print(f"[READ] loaded ys:\n{db_loaded.get_ys()}")
    
    # 5) Convert JSON database file to Excel
    db_loaded.json_to_excel(str(db_json_path), str(db_excel_path))
    print(f"[EXCEL] database exported to: {db_excel_path}")


if __name__ == "__main__":
    main()
