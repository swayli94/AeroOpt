import os
import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsOptimization, SettingsProblem
from AeroOpt.optimization.base import OptBaseFramework


class _DummyOpt(OptBaseFramework):
    def initialize_population(self) -> None:
        self.iteration = 1

    def generate_candidate_individuals(self) -> None:
        return None

    def select_valid_elite_from_total(self) -> None:
        self.db_valid = Database(self.problem, database_type="valid")
        self.db_elite = Database(self.problem, database_type="elite")


def _make_opt(problem, optimization_settings):
    opt = object.__new__(_DummyOpt)
    opt.problem = problem
    opt.optimization_settings = optimization_settings
    opt.user_func = _user_func
    opt.mp_evaluation = None
    opt.pre_process = None
    opt.post_process = None
    opt.iteration = 0
    opt.db_total = Database(problem, database_type="total")
    opt.db_valid = Database(problem, database_type="valid")
    opt.db_elite = Database(problem, database_type="elite")
    opt.db_candidate = Database(problem, database_type="population")
    return opt


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
    s.max_iterations = 2
    s.working_directory = str(tmp_path)
    return s


def _user_func(x):
    return True, np.array([x[0] * 0.5])


def _append_direct(db: Database, indi: Individual):
    db.individuals.append(indi)
    db.update_id_list()
    db._sorted = False


def test_base_properties(problem, optimization_settings):
    opt = _make_opt(problem, optimization_settings)
    assert opt.population_size == 4
    assert opt.max_iterations == 2
    assert opt.name == f"{optimization_settings.name}{problem.name}"
    assert opt.dir_save.endswith("Calculation")
    assert opt.dir_summary.endswith("Summary")
    assert opt.dir_runfiles.endswith("Runfiles")


def test_assign_id_to_candidate_individuals(problem, optimization_settings):
    opt = _make_opt(problem, optimization_settings)
    _append_direct(opt.db_total, Individual(problem, x=np.array([0.2]), y=np.array([0.1]), ID=10))

    opt.db_candidate = Database(problem, database_type="population")
    for x in [0.4, 0.6]:
        _append_direct(opt.db_candidate, Individual(problem, x=np.array([x]), y=np.array([x * 0.5]), ID=None))

    opt._assign_ID_to_candidate_individuals()
    ids = [indi.ID for indi in opt.db_candidate.individuals]
    assert ids == [11, 12]


def test_evaluate_db_candidate_and_merge_total(problem, optimization_settings, monkeypatch):
    opt = _make_opt(problem, optimization_settings)
    opt.db_candidate = Database(problem, database_type="population")
    _append_direct(opt.db_candidate, Individual(problem, x=np.array([0.3]), ID=1))
    _append_direct(opt.db_candidate, Individual(problem, x=np.array([0.7]), ID=2))

    def _patched_add_individual(self, indi, **kwargs):
        self.individuals.append(indi)
        self.update_id_list()
        self._sorted = False
        return True

    monkeypatch.setattr(Database, "add_individual", _patched_add_individual)
    opt.evaluate_db_candidate()

    assert opt.db_total.size >= 2
    ys = [indi.y for indi in opt.db_total.individuals if indi.y is not None]
    assert len(ys) >= 2
    assert any(np.allclose(y, np.array([0.15])) for y in ys)
    assert any(np.allclose(y, np.array([0.35])) for y in ys)
