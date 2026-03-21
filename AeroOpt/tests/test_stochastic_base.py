import os

import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsOptimization, SettingsProblem
from AeroOpt.optimization.stochastic.base import EvolutionaryAlgorithm, OptEvolutionaryFramework


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
def optimization_settings(settings_path, tmp_path):
    s = SettingsOptimization("default", fname_settings=settings_path)
    s.population_size = 8
    s.max_iterations = 2
    s.working_directory = str(tmp_path)
    return s


def test_evolutionary_pareto_dominance():
    # Vectors are in unified (all-minimize) space where larger values are better along each axis.
    assert EvolutionaryAlgorithm.pareto_dominance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0
    assert EvolutionaryAlgorithm.pareto_dominance(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == 1
    assert EvolutionaryAlgorithm.pareto_dominance(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == -1
    assert EvolutionaryAlgorithm.pareto_dominance(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 9


def test_evolutionary_get_unified_objectives_empty(problem):
    db = Database(problem, database_type="valid")
    ys = EvolutionaryAlgorithm._get_unified_objectives(db)
    assert ys.shape == (0, 0)


def test_evolutionary_get_unified_objectives_minimize(problem):
    db = Database(problem, database_type="valid")
    indi = Individual(problem, x=np.array([0.5]), y=np.array([0.4]), ID=1)
    db.add_individual(indi)
    ys = EvolutionaryAlgorithm._get_unified_objectives(db)
    assert ys.shape == (1, 1)
    assert np.allclose(ys[0, 0], -0.4)


def test_faster_non_dominated_ranking_single_objective(problem):
    db = Database(problem, database_type="valid")
    db.add_individual(Individual(problem, x=np.array([0.2]), y=np.array([0.8]), ID=1))
    db.add_individual(Individual(problem, x=np.array([0.3]), y=np.array([0.2]), ID=2))
    fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(db, is_valid_database=False)
    assert len(fronts) >= 1
    assert db.individuals[1].pareto_rank == 1
    assert db.individuals[0].pareto_rank == 2


def test_select_population_indices_respects_crowding():
    class _P:
        pass

    class _Indi:
        def __init__(self, cd):
            self.crowding_distance = cd

    db = type("Db", (), {})()
    db.individuals = [_Indi(0.1), _Indi(0.5), _Indi(0.2)]
    fronts = [[0, 1, 2]]
    out = EvolutionaryAlgorithm.select_population_indices(db, fronts, population_size=2)
    assert set(out) == {1, 2}


def test_opt_evolutionary_select_elite_empty_valid(problem, optimization_settings):
    opt = OptEvolutionaryFramework(problem=problem, optimization_settings=optimization_settings)
    opt.db_valid = Database(problem, database_type="valid")
    opt.select_elite_from_valid()
    assert opt.db_elite.size == 0


def test_opt_evolutionary_select_elite_from_valid(problem, optimization_settings):
    opt = OptEvolutionaryFramework(problem=problem, optimization_settings=optimization_settings)
    opt.db_valid = Database(problem, database_type="valid")
    opt.db_valid.add_individual(Individual(problem, x=np.array([0.2]), y=np.array([0.3]), ID=1))
    opt.db_valid.add_individual(Individual(problem, x=np.array([0.4]), y=np.array([0.5]), ID=2))
    opt.select_elite_from_valid()
    assert opt.db_elite.size >= 1
    assert all(indi.pareto_rank == 1 for indi in opt.db_elite.individuals)


def test_evolutionary_algorithm_str_repr():
    ea = EvolutionaryAlgorithm("NSGA-II")
    assert str(ea) == "NSGA-II"
    assert repr(ea) == "NSGA-II"
