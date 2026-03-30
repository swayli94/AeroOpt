import os

import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from AeroOpt.optimization import (
    DiffEvolution,
    DominanceBasedAlgorithm,
    OptDE,
    SettingsDE,
    SettingsOptimization,
)
from AeroOpt.optimization.utils import sample_de_rand_1_indices


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
    s.population_size = 4
    s.max_iterations = 2
    s.working_directory = str(tmp_path)
    return s


def _indi(problem: Problem, x: float, y: float, ID: int) -> Individual:
    indi = Individual(problem, x=np.array([x]), y=np.array([y]), ID=ID)
    indi.eval_constraints()
    return indi


def test_sample_de_rand_1_indices_single_population():
    rng = np.random.default_rng(0)
    assert sample_de_rand_1_indices(rng, n_pop=1, i_target=0) == (0, 0, 0)


def test_generate_candidate_individuals_requires_valid_population(problem):
    db_valid = Database(problem, database_type="valid")
    db_candidate = Database(problem, database_type="population")
    with pytest.raises(RuntimeError, match="No valid individuals"):
        DiffEvolution.generate_candidate_individuals(
            db_valid,
            db_candidate,
            population_size=4,
            iteration=1,
            scale_factor=0.5,
            cross_rate=0.9,
            rng=np.random.default_rng(1),
        )


def test_generate_candidate_individuals_builds_trials(problem):
    rng = np.random.default_rng(42)
    db_valid = Database(problem, database_type="valid")
    for i, x in enumerate([0.05, 0.35, 0.65, 0.92], start=1):
        db_valid.add_individual(
            _indi(problem, x, x * 0.5, i),
            check_duplication=False,
            print_warning_info=False,
        )
    db_valid._is_valid_database = True
    DominanceBasedAlgorithm.rank_pareto(db_valid)

    db_candidate = Database(problem, database_type="population")
    DiffEvolution.generate_candidate_individuals(
        db_valid,
        db_candidate,
        population_size=4,
        iteration=3,
        scale_factor=0.5,
        cross_rate=0.9,
        rng=rng,
    )
    assert db_candidate.size == 4
    for indi in db_candidate.individuals:
        assert indi.source == 'evolutionary_operator'
        assert indi.generation == 3
        assert problem.check_bounds_x(indi.x)


def test_opt_de_select_elite(problem, optimization_settings, settings_path):
    algo = SettingsDE("default", fname_settings=settings_path)
    opt = OptDE(
        problem=problem,
        optimization_settings=optimization_settings,
        algorithm_settings=algo,
    )
    opt.db_valid = Database(problem, database_type="valid")
    opt.db_valid.add_individual(
        Individual(problem, x=np.array([0.2]), y=np.array([0.3]), ID=1),
        check_duplication=False,
        print_warning_info=False,
    )
    opt.db_valid._is_valid_database = True
    opt.select_elite_from_valid()
    assert opt.db_elite.size >= 1
