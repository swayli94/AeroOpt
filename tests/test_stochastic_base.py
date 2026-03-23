import os

import numpy as np
import pytest

from AeroOpt.core import (
    Database, Individual, Problem, SettingsData, SettingsProblem
)
from AeroOpt.optimization import (
    DominanceBasedAlgorithm,
    OptNSGAII,
    SettingsNSGAII,
    SettingsOptimization,
)


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "AeroOpt", "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def optimization_settings(settings_path, tmp_path):
    s = SettingsOptimization("default", fname_settings=settings_path)
    s.population_size = 8
    s.max_iterations = 2
    s.working_directory = str(tmp_path)
    return s


def test_dominance_check_pareto_dominance():
    assert DominanceBasedAlgorithm.check_pareto_dominance(
        np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0
    assert DominanceBasedAlgorithm.check_pareto_dominance(
        np.array([2.0, 2.0]), np.array([1.0, 1.0])) == 1
    assert DominanceBasedAlgorithm.check_pareto_dominance(
        np.array([1.0, 1.0]), np.array([2.0, 2.0])) == -1
    assert DominanceBasedAlgorithm.check_pareto_dominance(
        np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 9


def test_get_unified_objectives_empty(problem):
    db = Database(problem, database_type="valid")
    ys = db.get_unified_objectives(scale=False)
    assert ys.shape == (0, 0)


def test_get_unified_objectives_minimize(problem):
    db = Database(problem, database_type="valid")
    indi = Individual(problem, x=np.array([0.5]), y=np.array([0.4]), ID=1)
    db.add_individual(indi)
    ys = db.get_unified_objectives(scale=False)
    assert ys.shape == (1, 1)
    assert np.allclose(ys[0, 0], -0.4)


def test_non_dominated_ranking_single_objective(problem):
    db = Database(problem, database_type="valid")
    db.add_individual(Individual(problem, x=np.array([0.2]), y=np.array([0.8]), ID=1))
    db.add_individual(Individual(problem, x=np.array([0.3]), y=np.array([0.2]), ID=2))
    fronts = DominanceBasedAlgorithm.non_dominated_ranking(db)
    assert len(fronts) >= 1
    assert db.individuals[1].pareto_rank == 1
    assert db.individuals[0].pareto_rank == 2


def test_rank_pareto_remaps_front_indices_after_sort(problem):
    db = Database(problem, database_type="valid")
    for i, (x, y) in enumerate(
            [(0.2, 0.4), (0.3, 0.1), (0.5, 0.2), (0.8, 0.35)], start=1):
        indi = Individual(problem, x=np.array([x]), y=np.array([y]), ID=i)
        indi.eval_constraints()
        db.add_individual(indi, check_duplication=False, print_warning_info=False)
    db._is_valid_database = True
    DominanceBasedAlgorithm.rank_pareto(db)
    assert db._sorted
    best_id = 2
    front0 = db._index_pareto_fronts[0]
    assert front0
    assert all(0 <= j < db.size for j in front0)
    assert {db.individuals[j].ID for j in front0} == {best_id}
    picked = DominanceBasedAlgorithm.select_parent_indices(db, n_select=1)
    assert len(picked) == 1
    assert db.individuals[picked[0]].ID == best_id


def test_select_parent_indices_respects_crowding():
    class _P:
        pass

    class _Indi:
        def __init__(self, cd):
            self.crowding_distance = cd

    db = type("Db", (), {})()
    db.individuals = [_Indi(0.1), _Indi(0.5), _Indi(0.2)]
    db._index_pareto_fronts = [[0, 1, 2]]
    out = DominanceBasedAlgorithm.select_parent_indices(db, n_select=2)
    assert set(out) == {1, 2}


def test_opt_nsgaii_select_elite_empty_valid(problem, optimization_settings, settings_path):
    algo = SettingsNSGAII("default", fname_settings=settings_path)
    opt = OptNSGAII(
        problem=problem,
        optimization_settings=optimization_settings,
        algorithm_settings=algo,
    )
    opt.db_valid = Database(problem, database_type="valid")
    opt.select_elite_from_valid()
    assert opt.db_elite.size == 0


def test_opt_nsgaii_select_elite_from_valid(problem, optimization_settings, settings_path):
    algo = SettingsNSGAII("default", fname_settings=settings_path)
    opt = OptNSGAII(
        problem=problem,
        optimization_settings=optimization_settings,
        algorithm_settings=algo,
    )
    opt.db_valid = Database(problem, database_type="valid")
    opt.db_valid.add_individual(Individual(problem, x=np.array([0.2]), y=np.array([0.3]), ID=1))
    opt.db_valid.add_individual(Individual(problem, x=np.array([0.4]), y=np.array([0.5]), ID=2))
    opt.select_elite_from_valid()
    assert opt.db_elite.size >= 1
    assert all(indi.pareto_rank == 1 for indi in opt.db_elite.individuals)


def test_copy_from_database_requires_same_problem_instance(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    p_a = Problem(sd, sp)
    p_b = Problem(sd, sp)
    db_a = Database(p_a, database_type="population")
    db_b = Database(p_b, database_type="population")
    db_b.add_individual(Individual(p_b, x=np.array([0.5]), y=np.array([0.2]), ID=1))
    with pytest.raises(ValueError, match="same problem"):
        db_a.copy_from_database(db_b)
