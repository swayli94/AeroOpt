import json
import os
import random

import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from AeroOpt.optimization import (
    SettingsNSGAIII,
    EvolutionaryAlgorithm,
    NSGAIII,
)


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def problem_biobj(tmp_path):
    """双目标问题（两个最小化目标），用于 NSGA-III 参考点选择路径。"""
    cfg = {
        "sd_bio": {
            "type": "SettingsData",
            "name": "bio",
            "name_input": ["x"],
            "name_output": ["f1", "f2"],
            "input_low": [0.0],
            "input_upp": [1.0],
            "input_precision": [1.0e-6],
            "output_low": [-1.0e6, -1.0e6],
            "output_upp": [1.0e6, 1.0e6],
            "output_precision": [0.0, 0.0],
            "critical_scaled_distance": 1.0e-6,
        },
        "sp_bio": {
            "type": "SettingsProblem",
            "name": "bio",
            "name_data_settings": "bio",
            "output_type": [-1, -1],
            "constraint_strings": [],
        },
    }
    path = os.path.join(tmp_path, "bio_settings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    sd = SettingsData("bio", fname_settings=path)
    sp = SettingsProblem("bio", sd, fname_settings=path)
    return Problem(sd, sp)


def _indi(problem: Problem, x: float, y, ID: int) -> Individual:
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    indi = Individual(problem, x=np.array([x]), y=y_arr, ID=ID)
    indi.eval_constraints()
    return indi


def _fill(db: Database, individuals):
    db.individuals.extend(individuals)
    db.update_id_list()
    db._sorted = False


def test_settings_nsgaiii_reads_template(settings_path):
    s = SettingsNSGAIII("default", fname_settings=settings_path)
    assert s.name == "default"
    assert s.cross_rate == pytest.approx(0.8)
    assert s.mut_rate == pytest.approx(0.8)
    assert s.n_partitions is None


def test_nsgaiii_algorithm_str_repr(settings_path):
    ea = NSGAIII("default", fname_settings=settings_path)
    assert str(ea) == "NSGAIII"
    assert repr(ea) == "NSGAIII"


def test_suggest_n_partitions():
    import math

    assert NSGAIII.suggest_n_partitions(1, 10) == 1
    # M=2: n_ref = p + 1; population_size=4 -> best p = 3 (four reference directions)
    assert NSGAIII.suggest_n_partitions(2, 4) == 3
    assert math.comb(NSGAIII.suggest_n_partitions(2, 4) + 1, 1) == 4


def test_das_dennis_reference_points():
    ref = NSGAIII.das_dennis_reference_points(1, 5)
    assert ref.shape == (1, 1)
    assert ref[0, 0] == pytest.approx(1.0)

    ref2 = NSGAIII.das_dennis_reference_points(2, 2)
    assert ref2.shape[1] == 2
    assert np.allclose(ref2.sum(axis=1), 1.0)


def test_normalize_objectives_nsgaiii():
    Z = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]], dtype=float)
    Zn = NSGAIII.normalize_objectives_nsgaiii(Z)
    assert Zn.shape == Z.shape
    assert np.all(Zn >= 0.0)


def test_build_temporary_parent_database_empty_raises(problem):
    db = Database(problem, database_type="valid")
    with pytest.raises(ValueError, match="empty valid"):
        NSGAIII.build_temporary_parent_database(db, population_size=4)


def test_generate_candidate_individuals_requires_valid_population(problem):
    db_valid = Database(problem, database_type="valid")
    db_candidate = Database(problem, database_type="population")
    with pytest.raises(RuntimeError, match="No valid individuals"):
        NSGAIII.generate_candidate_individuals(
            db_valid, db_candidate, population_size=4, iteration=1
        )


def test_generate_candidate_individuals_builds_offspring(problem):
    random.seed(123)
    np.random.seed(123)
    db_valid = Database(problem, database_type="valid")
    for i, x in enumerate([0.05, 0.35, 0.65, 0.92], start=1):
        db_valid.add_individual(
            _indi(problem, x, x * 0.5, i),
            check_duplication=False,
            print_warning_info=False,
        )

    EvolutionaryAlgorithm.rank_pareto(db_valid, is_valid_database=True)

    db_candidate = Database(problem, database_type="population")
    NSGAIII.generate_candidate_individuals(
        db_valid,
        db_candidate,
        population_size=4,
        iteration=2,
        cross_rate=1.0,
        pow_sbx=20.0,
        mut_rate=1.0,
        pow_poly=20.0,
    )

    assert db_candidate.size == 4
    for indi in db_candidate.individuals:
        assert indi.source == "GA"
        assert indi.generation == 2
        assert problem.check_bounds_x(indi.x)


def test_environmental_selection_two_objectives(problem_biobj):
    db = Database(problem_biobj, database_type="valid")
    pts = [
        ([0.1], [0.9, 0.1]),
        ([0.2], [0.7, 0.2]),
        ([0.3], [0.5, 0.4]),
        ([0.4], [0.4, 0.5]),
        ([0.5], [0.2, 0.7]),
        ([0.6], [0.1, 0.85]),
    ]
    for k, (xv, yv) in enumerate(pts, start=1):
        db.add_individual(
            _indi(problem_biobj, float(xv[0]), yv, k),
            check_duplication=False,
            print_warning_info=False,
        )

    idx = NSGAIII.environmental_selection(db, population_size=4, n_partitions=2)
    assert len(idx) == 4
    assert len(set(idx)) == 4
    assert all(0 <= i < db.size for i in idx)
