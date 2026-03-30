import os
from types import SimpleNamespace

import numpy as np
import pytest

from aeroopt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from aeroopt.optimization import SettingsOptimization
from aeroopt.optimization.hybrid.sbo import (
    PostProcessSBO,
    SBO,
    _surrogate_user_func,
)
from aeroopt.utils.surrogate import SurrogateModel


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
def optimization_settings(settings_path, tmp_path):
    s = SettingsOptimization("default", fname_settings=settings_path)
    s.population_size = 4
    s.max_iterations = 1
    s.working_directory = str(tmp_path)
    return s


class _StubSurrogate(SurrogateModel):
    def train(self, xs, ys, **kwargs):
        self._last_train = (np.asarray(xs).copy(), np.asarray(ys).copy())
        self._model = object()
        self._size = xs.shape[0]

    def predict(self, xs, **kwargs):
        xs = np.asarray(xs, dtype=float)
        n = xs.shape[0]
        col = np.sum(xs, axis=1, keepdims=True)
        return np.broadcast_to(col, (n, self.problem.n_output)).copy()

    def full_predict(self, xs, **kwargs):
        ys = self.predict(xs, **kwargs)
        return {"ys": ys, "epistemic_variance": np.zeros_like(ys)}

    def predict_for_adaptive_sampling(self, xs, **kwargs):
        return self.predict(xs, **kwargs)

    def evaluate_performance(self, xs, ys_actual, **kwargs):
        self._last_perf_args = (
            np.asarray(xs).copy(),
            np.asarray(ys_actual).copy(),
        )
        return {"metric": 0.42, "RMSE": np.array([0.1])}


class _InnerOptShell:
    def __init__(self, problem, n_individuals: int = 4):
        self.problem = problem
        self.db_total = Database(problem, database_type="total")
        self.db_valid = Database(problem, database_type="valid")
        self.db_elite = Database(problem, database_type="elite")
        self.db_candidate = Database(problem, database_type="population")
        self.iteration = 0
        self.user_func = None
        self.user_func_supports_parallel = False
        self._n_individuals = n_individuals

    def initialize(self) -> None:
        self.db_total.empty_database()
        self.db_valid.empty_database()
        self.db_elite.empty_database()
        self.db_candidate.empty_database()
        self.iteration = 0

    def main(self):
        self.initialize()
        rng = np.random.default_rng(42)
        for i in range(self._n_individuals):
            xv = np.array([0.15 + 0.1 * i])
            yv = np.array([0.25 + 0.05 * i])
            indi = Individual(problem=self.problem, x=xv, y=yv, ID=i + 1)
            self.db_total.add_individual(
                indi,
                check_duplication=False,
                check_bounds=True,
                deepcopy=False,
                print_warning_info=False,
            )
        self.db_valid.copy_from_database(self.db_total, deepcopy=True)
        self.db_valid._is_valid_database = True


def test_surrogate_user_func_returns_parallel_batch(problem):
    sur = _StubSurrogate(problem)
    xs = np.array([[0.2], [0.3]])
    ok, ys = _surrogate_user_func(xs, surrogate=sur)
    assert ok == [True, True]
    assert ys.shape == (2, problem.n_output)
    np.testing.assert_allclose(ys[:, 0], [0.2, 0.3])


def test_post_process_sbo_calls_evaluate_performance_with_sliced_outputs(
    problem, optimization_settings,
):
    surrogate = _StubSurrogate(problem)
    surrogate.train(
        np.ones((2, problem.n_input)),
        np.ones((2, problem.n_output)),
    )
    db_c = Database(problem, database_type="population")
    db_c.add_individual(
        Individual(problem, x=np.array([0.1]), y=np.array([0.9]), ID=1),
        check_duplication=False,
        print_warning_info=False,
    )
    db_c.add_individual(
        Individual(problem, x=np.array([0.2]), y=np.array([0.8]), ID=2),
        check_duplication=False,
        print_warning_info=False,
    )

    opt = SimpleNamespace(
        db_candidate=db_c,
        index_outputs_for_surrogate=np.array([0], dtype=int),
        log=lambda *a, **k: None,
    )
    pp = PostProcessSBO(opt, surrogate)
    pp.apply()

    assert surrogate._last_perf_args is not None
    xs_arg, ys_arg = surrogate._last_perf_args
    assert xs_arg.shape == (2, problem.n_input)
    np.testing.assert_allclose(ys_arg[:, 0], [0.9, 0.8])


def test_sbo_output_indices_match_problem(problem, optimization_settings):
    surrogate = _StubSurrogate(problem)
    inner = _InnerOptShell(problem)
    sbo = SBO(
        problem,
        optimization_settings,
        surrogate,
        inner,
        user_func=None,
        mp_evaluation=None,
    )
    assert sbo.outputs_for_surrogate == list(problem.name_output)
    np.testing.assert_array_equal(
        sbo.index_outputs_for_surrogate,
        np.arange(problem.n_output, dtype=int),
    )


def test_sbo_raises_when_surrogate_outputs_not_in_problem(
    problem, optimization_settings,
):
    surrogate = _StubSurrogate(problem)
    surrogate.problem = SimpleNamespace(name_output=["nonexistent_output"])
    inner = _InnerOptShell(problem)
    with pytest.raises(Exception, match="Surrogate model problem error"):
        SBO(
            problem,
            optimization_settings,
            surrogate,
            inner,
            user_func=None,
            mp_evaluation=None,
        )


def test_sbo_update_parameters_trains_on_valid_slice(problem, optimization_settings):
    surrogate = _StubSurrogate(problem)
    inner = _InnerOptShell(problem)
    sbo = SBO(
        problem,
        optimization_settings,
        surrogate,
        inner,
        user_func=None,
        mp_evaluation=None,
    )
    sbo.db_valid.empty_database()
    for i, (xv, yv) in enumerate([([0.1], [0.2]), ([0.3], [0.4])], start=1):
        indi = Individual(problem, x=np.array(xv), y=np.array(yv), ID=i)
        sbo.db_valid.add_individual(
            indi,
            check_duplication=False,
            print_warning_info=False,
        )
    sbo.db_valid._is_valid_database = True

    sbo.update_parameters()

    assert surrogate._last_train is not None
    tx, ty = surrogate._last_train
    assert tx.shape == (2, problem.n_input)
    assert ty.shape == (2, len(sbo.index_outputs_for_surrogate))
    np.testing.assert_allclose(ty[:, 0], [0.2, 0.4])


def test_sbo_generate_candidate_from_inner_total(problem, optimization_settings):
    surrogate = _StubSurrogate(problem)
    inner = _InnerOptShell(problem, n_individuals=4)
    sbo = SBO(
        problem,
        optimization_settings,
        surrogate,
        inner,
        user_func=None,
        mp_evaluation=None,
    )
    sbo.iteration = 2
    sbo.db_candidate.add_individual(
        Individual(problem, x=np.array([0.99]), y=np.array([0.1]), ID=99),
        check_duplication=False,
        print_warning_info=False,
    )

    sbo.generate_candidate_individuals()

    assert sbo.db_candidate.size == 4
    assert all(getattr(indi, "source", None) == "surrogate_prediction" for indi in sbo.db_candidate.individuals)
    assert all(indi.generation == 2 for indi in sbo.db_candidate.individuals)
    for indi in sbo.db_candidate.individuals:
        assert hasattr(indi, "_y_predicted")
        assert indi._y_predicted.shape == (problem.n_output,)


def test_sbo_select_elite_delegates_with_empty_valid(problem, optimization_settings):
    surrogate = _StubSurrogate(problem)
    inner = _InnerOptShell(problem)
    sbo = SBO(
        problem,
        optimization_settings,
        surrogate,
        inner,
        user_func=None,
        mp_evaluation=None,
    )
    sbo.db_valid.empty_database()
    sbo.db_elite.empty_database()
    sbo.select_elite_from_valid()
    assert sbo.db_elite.size == 0
