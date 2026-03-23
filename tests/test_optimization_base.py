import os
import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from AeroOpt.optimization import (
    OptBaseFramework, SettingsOptimization, PreProcess
)


class _DummyOpt(OptBaseFramework):
    '''
    Dummy optimization framework for testing.
    '''
    def initialize_population(self) -> None:
        self.iteration = 1

    def generate_candidate_individuals(self) -> None:
        return None

    def select_elite_from_valid(self) -> None:
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
    opt.analyze_total = None
    opt.analyze_valid = None
    opt.log = lambda *args, **kwargs: None
    return opt


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
    assert opt.name == f"{optimization_settings.name}-{problem.name}"
    assert opt.dir_save.endswith("Calculation")
    assert opt.dir_summary.endswith("Summary")
    assert opt.dir_runfiles.endswith("Runfiles")


def test_max_id_zero_when_total_empty(problem, optimization_settings):
    opt = _make_opt(problem, optimization_settings)
    assert opt.db_total.size == 0
    assert opt.max_ID == 0


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
        return True, ""

    monkeypatch.setattr(Database, "add_individual", _patched_add_individual)
    opt.evaluate_db_candidate()
    opt.update_total_and_valid_with_candidate()

    assert opt.db_total.size >= 2
    ys = [indi.y for indi in opt.db_total.individuals if indi.y is not None]
    assert len(ys) >= 2
    assert any(np.allclose(y, np.array([0.15])) for y in ys)
    assert any(np.allclose(y, np.array([0.35])) for y in ys)


class _DummyProblemForPreProcess:
    def __init__(self):
        self.name = "DummyPreProcessProblem"
        self.critical_scaled_distance = 0.2
        self.calculation_folder = ""
        self._lb = 0.0
        self._ub = 1.0

    def scale_x(self, x, reverse=False):
        # Use identity transform so expected values are easy to verify.
        return np.asarray(x, dtype=float).copy()

    def apply_bounds_x(self, x):
        np.clip(x, self._lb, self._ub, out=x)

    def check_bounds_x(self, x):
        x = np.asarray(x, dtype=float)
        return np.all(x >= self._lb) and np.all(x <= self._ub)


class _DummyAnalyzeValid:
    def __init__(self, distance_matrix, valid_xs):
        self._distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.database = type("Db", (), {})()
        self.database.individuals = [
            type("I", (), {"x": np.asarray([vx], dtype=float), "ID": 100 + k})()
            for k, vx in enumerate(valid_xs)
        ]

    def calculate_distance_to_database(self, scaled_xs, update_attributes=True):
        return self._distance_matrix


def _preprocess_dummy_opt(problem, analyze_valid=None, dir_save="tmp_dir"):
    opt = type("Opt", (), {})()
    opt.problem = problem
    opt.dir_save = dir_save
    opt.db_valid = type("DbV", (), {"size": 2})()
    opt.log = lambda *args, **kwargs: None
    if analyze_valid is not None:
        opt.analyze_valid = analyze_valid
    return opt


def test_preprocess_restrict_x_values_by_valid_database():
    problem = _DummyProblemForPreProcess()
    opt = _preprocess_dummy_opt(
        problem,
        analyze_valid=_DummyAnalyzeValid(
            distance_matrix=[[0.10, 0.60], [0.35, 0.70], [0.50, 0.10]],
            valid_xs=[0.20, 0.80],
        ),
    )
    pp = PreProcess(opt)

    xs = np.array([[0.00], [0.50], [1.20]], dtype=float)
    xs_new = pp._restrict_x_values_by_valid_database(
        xs, min_scaled_distance=0.05, max_scaled_distance=0.30
    )

    # 1) too close -> move to at least critical distance (0.2) from x_ref=0.2
    assert np.allclose(xs_new[0], np.array([0.0]))
    # 2) too far -> pull towards x_ref=0.2 to max distance 0.3 => 0.2 + 0.3*(0.3/0.35)
    assert np.allclose(xs_new[1], np.array([0.4571428571]))
    # 3) already in range [0.2, 0.3] around nearest valid -> unchanged by distance rule, then bounded
    assert np.allclose(xs_new[2], np.array([1.0]))


def test_preprocess_check_feasibility_sets_calculation_folder(monkeypatch):
    problem = _DummyProblemForPreProcess()
    opt = _preprocess_dummy_opt(problem, dir_save="run_dir")
    opt.mp_evaluation = None
    pp = PreProcess(opt)

    def _fake_evaluate(self, mp_evaluation=None, user_func=None):
        for indi in self.individuals:
            x0 = float(indi.x[0])
            indi.valid_evaluation = x0 >= 0.2
            indi.sum_violation = 0.0 if x0 <= 0.8 else 1.0

    def _fake_add_individual(self, indi, **kwargs):
        self.individuals.append(indi)
        self.update_id_list()
        self._sorted = False
        return True, ""

    monkeypatch.setattr(Database, "add_individual", _fake_add_individual)
    monkeypatch.setattr(Database, "evaluate_individuals", _fake_evaluate)
    xs = np.array([[0.1], [0.3], [0.9]], dtype=float)
    flags, id_list = pp._check_pre_processing_feasibility(xs, pre_processing_problem=problem)

    assert problem.calculation_folder == os.path.join("run_dir", "PreProcess")
    assert flags == [False, True, False]
    assert id_list == [1, 2, 3]


def test_preprocess_adjust_x_values_by_valid_database(monkeypatch):
    problem = _DummyProblemForPreProcess()
    opt = _preprocess_dummy_opt(
        problem,
        analyze_valid=_DummyAnalyzeValid(
            distance_matrix=[[0.12, 0.40], [0.20, 0.60]],
            valid_xs=[0.20, 0.80],
        ),
    )
    pp = PreProcess(opt)

    xs = np.array([[0.05], [0.60], [0.95]], dtype=float)
    feasibility_flags = np.array([False, True, False], dtype=bool)

    xs_adjusted = np.array([[0.00], [1.10]], dtype=float)

    def _fake_restrict(_xs, min_scaled_distance=0.01, max_scaled_distance=0.10, ID_list=None):
        # Ensure only infeasible rows are sent for adjustment.
        assert np.allclose(_xs, np.array([[0.05], [0.95]]))
        return xs_adjusted

    monkeypatch.setattr(pp, "_restrict_x_values_by_valid_database", _fake_restrict)
    out = pp._adjust_x_values_by_valid_database(xs, feasibility_flags)

    assert np.allclose(out, np.array([[0.00], [0.60], [1.10]]))


def test_preprocess_adjust_x_feasibility_length_mismatch_raises():
    problem = _DummyProblemForPreProcess()
    opt = _preprocess_dummy_opt(problem)
    pp = PreProcess(opt)
    xs = np.array([[0.1], [0.2]], dtype=float)
    with pytest.raises(ValueError, match="Length mismatch"):
        pp._adjust_x_values_by_valid_database(xs, [True, True, False])


class _AnalyzeStub:
    def __init__(self, database):
        self.database = database


def test_update_total_and_valid_keeps_db_valid_instance_and_analyze_pointers(
    problem, optimization_settings,
):
    """In-place merge + copy_from_database: same db_total/db_valid objects, analyzers stay valid."""
    opt = _make_opt(problem, optimization_settings)
    opt.analyze_total = _AnalyzeStub(opt.db_total)
    opt.analyze_valid = _AnalyzeStub(opt.db_valid)

    id_total_before = id(opt.db_total)
    id_valid_before = id(opt.db_valid)

    opt.db_candidate = Database(problem, database_type="population")
    cand = Individual(problem, x=np.array([0.55]), ID=5)
    cand.valid_evaluation = True
    _append_direct(opt.db_candidate, cand)

    opt.update_total_and_valid_with_candidate()

    assert id(opt.db_total) == id_total_before
    assert id(opt.db_valid) == id_valid_before
    assert opt.analyze_total.database is opt.db_total
    assert opt.analyze_valid.database is opt.db_valid
    assert opt.db_total.size == 1
    assert opt.analyze_total.database.size == 1
    assert opt.analyze_valid.database.size == 1


def test_update_total_and_valid_repairs_mispointed_analyze_wrong_database(
    problem, optimization_settings,
):
    """If analyze_* pointed at another Database, they are re-bound to db_total / db_valid."""
    opt = _make_opt(problem, optimization_settings)
    wrong_db = Database(problem, database_type="population")
    opt.analyze_total = _AnalyzeStub(wrong_db)
    opt.analyze_valid = _AnalyzeStub(wrong_db)

    opt.db_candidate = Database(problem, database_type="population")
    cand = Individual(problem, x=np.array([0.33]), ID=7)
    cand.valid_evaluation = True
    _append_direct(opt.db_candidate, cand)

    opt.update_total_and_valid_with_candidate()

    assert opt.analyze_total.database is opt.db_total
    assert opt.analyze_valid.database is opt.db_valid
    assert opt.analyze_total.database is not wrong_db
    assert opt.analyze_valid.database is not wrong_db
    assert opt.db_total.size == 1
    assert wrong_db.size == 0


def test_update_total_and_valid_analyze_none_does_not_raise(problem, optimization_settings):
    opt = _make_opt(problem, optimization_settings)
    opt.analyze_total = None
    opt.analyze_valid = None

    opt.db_candidate = Database(problem, database_type="population")
    cand = Individual(problem, x=np.array([0.44]), ID=8)
    cand.valid_evaluation = True
    _append_direct(opt.db_candidate, cand)

    opt.update_total_and_valid_with_candidate()
    assert opt.db_total.size == 1


def test_update_total_and_valid_repairs_analyze_valid_only_when_total_ok(
    problem, optimization_settings,
):
    opt = _make_opt(problem, optimization_settings)
    opt.analyze_total = _AnalyzeStub(opt.db_total)
    wrong_db = Database(problem, database_type="population")
    opt.analyze_valid = _AnalyzeStub(wrong_db)

    opt.db_candidate = Database(problem, database_type="population")
    cand = Individual(problem, x=np.array([0.66]), ID=9)
    cand.valid_evaluation = True
    _append_direct(opt.db_candidate, cand)

    opt.update_total_and_valid_with_candidate()

    assert opt.analyze_total.database is opt.db_total
    assert opt.analyze_valid.database is opt.db_valid


def test_update_total_and_valid_repairs_analyze_total_only_when_valid_ok(
    problem, optimization_settings,
):
    opt = _make_opt(problem, optimization_settings)
    wrong_db = Database(problem, database_type="population")
    opt.analyze_total = _AnalyzeStub(wrong_db)
    opt.analyze_valid = _AnalyzeStub(opt.db_valid)

    opt.db_candidate = Database(problem, database_type="population")
    cand = Individual(problem, x=np.array([0.77]), ID=10)
    cand.valid_evaluation = True
    _append_direct(opt.db_candidate, cand)

    opt.update_total_and_valid_with_candidate()

    assert opt.analyze_total.database is opt.db_total
    assert opt.analyze_valid.database is opt.db_valid
