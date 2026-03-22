import os
import random

import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsOptimization, SettingsProblem
from AeroOpt.optimization.stochastic.base import EvolutionaryAlgorithm
from AeroOpt.optimization.stochastic.nsgaii import NSGAII


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
    s.population_size = 4
    s.max_iterations = 3
    s.working_directory = str(tmp_path)
    return s


def _indi(problem: Problem, x: float, y: float, ID: int) -> Individual:
    indi = Individual(problem, x=np.array([x]), y=np.array([y]), ID=ID)
    indi.eval_constraints()
    return indi


def _fill(db: Database, individuals):
    db.individuals.extend(individuals)
    db.update_id_list()
    db._sorted = False


def test_pareto_dominance_cases():
    y = np.array([1.0, 1.0])
    y2 = np.array([2.0, 3.0])
    assert EvolutionaryAlgorithm.pareto_dominance(y, y2) == -1
    assert EvolutionaryAlgorithm.pareto_dominance(y2, y) == 1
    assert EvolutionaryAlgorithm.pareto_dominance(y, y.copy()) == 0
    assert EvolutionaryAlgorithm.pareto_dominance(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 9


def test_faster_non_dominated_ranking_and_select(problem):
    db = Database(problem, database_type="valid")
    _fill(
        db,
        [
            _indi(problem, 0.2, 0.1, 1),
            _indi(problem, 0.4, 0.3, 2),
            _indi(problem, 0.6, 0.5, 3),
            _indi(problem, 0.8, 0.7, 4),
        ],
    )

    fronts = EvolutionaryAlgorithm.faster_non_dominated_ranking(db, is_valid_database=True)
    EvolutionaryAlgorithm.assign_crowding_distance(db, fronts)
    selected = EvolutionaryAlgorithm.select_population_indices(db, fronts, population_size=2)

    assert fronts[0] == [0]
    assert db.individuals[0].pareto_rank == 1
    assert len(selected) == 2
    assert 0 in selected


def test_assign_crowding_distance_boundary_is_inf(problem):
    db = Database(problem, database_type="valid")
    _fill(
        db,
        [
            _indi(problem, 0.1, 0.05, 1),
            _indi(problem, 0.4, 0.30, 2),
            _indi(problem, 0.6, 0.45, 3),
            _indi(problem, 0.9, 0.80, 4),
        ],
    )
    fronts = [[0, 1, 2, 3]]
    EvolutionaryAlgorithm.assign_crowding_distance(db, fronts)
    assert np.isinf(db.individuals[0].crowding_distance)
    assert np.isinf(db.individuals[3].crowding_distance)
    assert db.individuals[1].crowding_distance >= 0.0
    assert db.individuals[2].crowding_distance >= 0.0


def test_binary_tournament_selection_empty_pool_raises(problem):
    pool = Database(problem, database_type="valid")
    with pytest.raises(ValueError, match="empty"):
        NSGAII.binary_tournament_selection(pool, n_select=2)


def test_binary_tournament_selection_size(problem):
    random.seed(10)
    pool = Database(problem, database_type="valid")
    inds = []
    for i, x in enumerate([0.2, 0.4, 0.6, 0.8], start=1):
        indi = _indi(problem, x, x * 0.5, i)
        indi.pareto_rank = i
        indi.crowding_distance = float(5 - i)
        inds.append(indi)
    _fill(pool, inds)
    pool.sort_database(sort_type=0)
    selected = NSGAII.binary_tournament_selection(pool, n_select=4)
    assert len(selected) == 4
    assert all(isinstance(indi, Individual) for indi in selected)


def test_sbx_crossover_and_polynomial_mutation_bounds(problem):
    random.seed(42)
    np.random.seed(42)
    p1 = np.array([0.2])
    p2 = np.array([0.8])
    c1, c2 = NSGAII.sbx_crossover(p1, p2, problem, cross_rate=1.0, pow_sbx=20.0)
    m1 = NSGAII.polynomial_mutation(c1, problem, mut_rate=1.0, pow_poly=20.0)
    m2 = NSGAII.polynomial_mutation(c2, problem, mut_rate=1.0, pow_poly=20.0)

    assert c1.shape == (1,)
    assert c2.shape == (1,)
    assert m1.shape == (1,)
    assert m2.shape == (1,)
    assert problem.check_bounds_x(c1)
    assert problem.check_bounds_x(c2)
    assert problem.check_bounds_x(m1)
    assert problem.check_bounds_x(m2)


def test_generate_candidate_individuals_requires_valid_population(problem):
    db_valid = Database(problem, database_type="valid")
    db_candidate = Database(problem, database_type="population")
    with pytest.raises(RuntimeError, match="No valid individuals"):
        NSGAII.generate_candidate_individuals(
            db_valid, db_candidate, population_size=4, iteration=1
        )


def test_generate_candidate_individuals_builds_offspring(problem):
    random.seed(123)
    np.random.seed(123)
    db_valid = Database(problem, database_type="valid")
    for i, x in enumerate([0.05, 0.35, 0.65, 0.92], start=1):
        db_valid.add_individual(_indi(problem, x, x * 0.5, i), check_duplication=False, print_warning_info=False)

    EvolutionaryAlgorithm.rank_pareto(db_valid, is_valid_database=True)

    db_candidate = Database(problem, database_type="population")
    NSGAII.generate_candidate_individuals(
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
