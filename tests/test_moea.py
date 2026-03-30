import json
import os

import numpy as np
import pytest

from aeroopt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from aeroopt.optimization.moea import (
    Algorithm,
    DecompositionBasedAlgorithm,
    DominanceBasedAlgorithm,
)


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "aeroopt", "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def problem_biobj(tmp_path):
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


def _indi(p: Problem, x: float, y, ID: int) -> Individual:
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    indi = Individual(p, x=np.array([x]), y=y_arr, ID=ID)
    indi.eval_constraints()
    return indi


def _append_valid(db: Database, individuals):
    db.individuals.extend(individuals)
    db.update_id_list()
    db._sorted = False
    db._is_valid_database = True


def test_algorithm_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Algorithm()


def test_non_dominated_ranking_empty_database(problem):
    db = Database(problem, database_type="valid")
    db._is_valid_database = True
    assert DominanceBasedAlgorithm.non_dominated_ranking(db) == []


def test_rank_pareto_noop_on_empty_database(problem):
    db = Database(problem, database_type="valid")
    DominanceBasedAlgorithm.rank_pareto(db)
    assert db.size == 0


def test_build_temporary_parent_database_requires_valid_flag(problem):
    db = Database(problem, database_type="valid")
    _append_valid(db, [_indi(problem, 0.2, 0.1, 1)])
    db._is_valid_database = False
    with pytest.raises(ValueError, match="valid database"):
        DominanceBasedAlgorithm.build_temporary_parent_database(db, population_size=4)


def test_build_temporary_parent_database_empty_raises(problem):
    db = Database(problem, database_type="valid")
    db._is_valid_database = True
    with pytest.raises(ValueError, match="empty valid"):
        DominanceBasedAlgorithm.build_temporary_parent_database(db, population_size=4)


def test_build_temporary_parent_database_returns_same_when_fits(problem_biobj):
    db = Database(problem_biobj, database_type="valid")
    _append_valid(
        db,
        [
            _indi(problem_biobj, 0.1, [0.5, 0.4], 1),
            _indi(problem_biobj, 0.2, [0.4, 0.5], 2),
        ],
    )
    parent = DominanceBasedAlgorithm.build_temporary_parent_database(db, population_size=10)
    assert parent is db
    assert parent.size == 2


def test_build_temporary_parent_database_truncates_by_crowding(problem_biobj):
    db = Database(problem_biobj, database_type="valid")
    _append_valid(
        db,
        [
            _indi(problem_biobj, 0.1, [0.2, 0.9], 1),
            _indi(problem_biobj, 0.2, [0.3, 0.7], 2),
            _indi(problem_biobj, 0.3, [0.9, 0.2], 3),
            _indi(problem_biobj, 0.4, [0.5, 0.5], 4),
        ],
    )
    parent = DominanceBasedAlgorithm.build_temporary_parent_database(db, population_size=2)
    assert parent is not db
    assert parent.size == 2


def test_select_elite_from_valid_empty(problem):
    db_v = Database(problem, database_type="valid")
    db_v._is_valid_database = True
    db_e = Database(problem, database_type="elite")
    DominanceBasedAlgorithm.select_elite_from_valid(db_v, db_e)
    assert db_e.size == 0


def test_select_elite_from_valid_copies_first_front(problem_biobj):
    db_v = Database(problem_biobj, database_type="valid")
    _append_valid(
        db_v,
        [
            _indi(problem_biobj, 0.1, [0.2, 0.3], 1),
            _indi(problem_biobj, 0.2, [0.7, 0.1], 2),
            _indi(problem_biobj, 0.3, [0.35, 0.4], 3),
        ],
    )
    db_e = Database(problem_biobj, database_type="elite")
    DominanceBasedAlgorithm.select_elite_from_valid(db_v, db_e)
    assert db_e.size == 2
    assert {indi.ID for indi in db_e.individuals} == {1, 2}


def test_assign_crowding_distance_two_member_front_all_inf(problem):
    db = Database(problem, database_type="valid")
    _append_valid(db, [_indi(problem, 0.1, 0.2, 1), _indi(problem, 0.5, 0.6, 2)])
    db._index_pareto_fronts = [[0, 1]]
    DominanceBasedAlgorithm.assign_crowding_distance(db)
    assert np.isinf(db.individuals[0].crowding_distance)
    assert np.isinf(db.individuals[1].crowding_distance)


def test_das_dennis_nonpositive_objectives():
    assert DecompositionBasedAlgorithm.das_dennis_reference_points(0, 3).shape == (0, 0)


def test_default_decomposition_name():
    assert DecompositionBasedAlgorithm.default_decomposition_name(1) == "tchebicheff"
    assert DecompositionBasedAlgorithm.default_decomposition_name(2) == "tchebicheff"
    assert DecompositionBasedAlgorithm.default_decomposition_name(3) == "pbi"


def test_decomposed_values_dimension_mismatch_raises():
    ys = np.array([[0.1, 0.2]])
    w = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="Objective dimension mismatch"):
        DecompositionBasedAlgorithm.decomposed_values(ys, w, np.array([0.0, 0.0]), "tchebicheff")

    with pytest.raises(ValueError, match="ideal length"):
        DecompositionBasedAlgorithm.decomposed_values(
            ys, np.array([[0.5, 0.5]]), np.array([0.0]), "tchebicheff")


def test_decomposed_values_unknown_method_raises():
    ys = np.array([[0.1, 0.2]])
    w = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="Unknown decomposition"):
        DecompositionBasedAlgorithm.decomposed_values(ys, w, np.array([0.0, 0.0]), "unknown")


def test_decomposed_values_pbi_smoke():
    ys = np.array([[0.3, 0.4], [0.5, 0.2]])
    w = np.array([[0.5, 0.5], [0.2, 0.8]])
    ideal = np.array([0.0, 0.0])
    g = DecompositionBasedAlgorithm.decomposed_values(ys, w, ideal, "pbi", pbi_theta=5.0)
    assert g.shape == (2, 2)
    assert np.all(np.isfinite(g))


def test_reference_direction_progress_non_2d_returns_empty():
    ordered, best, ref = DecompositionBasedAlgorithm.reference_direction_progress(
        np.array([0.1, 0.2]), n_partitions=3)
    assert ordered.size == 0
    assert best.size == 0
    assert ref.size == 0


def test_reference_direction_progress_single_objective_branch():
    ys = np.array([[0.2], [0.1], [0.3]])
    ordered, best, ref = DecompositionBasedAlgorithm.reference_direction_progress(ys, n_partitions=4)
    assert np.array_equal(ordered, np.array([0]))
    assert best.shape == (1,)
    assert ref.shape == (1, 1)


def test_reference_direction_progress_ideal_length_raises():
    ys = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="ideal length"):
        DecompositionBasedAlgorithm.reference_direction_progress(
            ys, n_partitions=2, ideal=np.array([0.0]))


def test_reference_direction_progress_bad_decomposition_raises():
    ys = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="decomposition must be"):
        DecompositionBasedAlgorithm.reference_direction_progress(
            ys, n_partitions=2, decomposition="invalid")


def test_reference_direction_progress_biobj_ordering():
    ys = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    ordered, best_g, ref = DecompositionBasedAlgorithm.reference_direction_progress(
        ys, n_partitions=2, pareto_front_only=True, decomposition="tchebicheff")
    assert ref.shape[1] == 2
    assert ordered.shape[0] == ref.shape[0]
    assert best_g.shape[0] == ref.shape[0]
    assert set(ordered.tolist()) == set(range(ref.shape[0]))


def test_pareto_first_front_mask():
    ys = np.array([[0.2, 0.8], [0.8, 0.2], [0.9, 0.9]])
    mask = DecompositionBasedAlgorithm._pareto_first_front_mask(ys)
    assert bool(mask[0])
    assert bool(mask[1])
    assert not bool(mask[2])


def test_find_slow_directions_empty_database(problem):
    db = Database(problem, database_type="valid")
    db._is_valid_database = True
    ordered, best, ref = DecompositionBasedAlgorithm.find_slow_directions(db, n_partitions=3)
    assert ordered.size == 0
    assert best.size == 0
    assert ref.shape[1] == problem.n_objective


def test_find_slow_directions_biobj(problem_biobj):
    db = Database(problem_biobj, database_type="valid")
    _append_valid(
        db,
        [
            _indi(problem_biobj, 0.1, [0.2, 0.7], 1),
            _indi(problem_biobj, 0.2, [0.7, 0.2], 2),
        ],
    )
    n_part = 2
    ordered, best, ref = DecompositionBasedAlgorithm.find_slow_directions(db, n_partitions=n_part)
    assert ref.shape[1] == 2
    assert ref.shape[0] == DecompositionBasedAlgorithm.das_dennis_reference_points(2, n_part).shape[0]
    assert ordered.shape[0] == ref.shape[0]
    assert best.shape[0] == ref.shape[0]
    assert np.all(np.isfinite(best))
