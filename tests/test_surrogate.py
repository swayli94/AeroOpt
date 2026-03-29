import os

import numpy as np
import pytest

from AeroOpt.core.problem import Problem
from AeroOpt.core.settings import SettingsData, SettingsProblem
from AeroOpt.utils.surrogate import Kriging, SurrogateModel


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "AeroOpt", "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


class _ConcreteSurrogate(SurrogateModel):
    '''Minimal concrete implementation for interface tests.'''

    def train(self, xs, ys, **kwargs):
        self._model = {"xs_mean": float(np.mean(xs)), "ys_mean": float(np.mean(ys))}
        self._size = int(xs.shape[0])

    def predict(self, xs, **kwargs):
        xs = np.asarray(xs, dtype=float)
        n = xs.shape[0]
        out = np.sum(xs, axis=1, keepdims=True)
        return np.broadcast_to(out, (n, self.problem.n_output)).copy()

    def full_predict(self, xs, **kwargs):
        ys = self.predict(xs, **kwargs)
        return {"ys": ys, "epistemic_variance": np.zeros_like(ys)}

    def predict_for_adaptive_sampling(self, xs, **kwargs):
        result = self.full_predict(xs, **kwargs)
        epistemic_std = np.sqrt(np.clip(result["epistemic_variance"], 0.0, None))
        ys = result["ys"]
        criteria = np.zeros_like(ys)
        otype = np.asarray(self.problem.problem_settings.output_type, dtype=int)
        for i in range(self.n_output):
            if otype[i] == 1:
                criteria[:, i] = ys[:, i] + epistemic_std[:, i]
            elif otype[i] == -1:
                criteria[:, i] = ys[:, i] - epistemic_std[:, i]
            else:
                criteria[:, i] = epistemic_std[:, i]
        return criteria

    def evaluate_performance(self, xs, ys_actual, **kwargs):
        pred = self.predict(xs)
        rmse = np.sqrt(np.mean((pred - ys_actual) ** 2, axis=0))
        mae = np.mean(np.abs(pred - ys_actual), axis=0)
        return {"RMSE": rmse, "MAE": mae}


def test_surrogate_model_cannot_instantiate_abstract(problem):
    with pytest.raises(TypeError):
        SurrogateModel(problem, model_name="x", train_on_scaled_data=True)


def test_concrete_surrogate_init_and_properties(problem):
    s = _ConcreteSurrogate(problem, model_name="test", train_on_scaled_data=False)
    assert s.model is None
    assert s.size == 0
    assert s.model_name == "test"
    assert s.train_on_scaled_data is False
    assert s.problem is problem
    assert s.n_input == problem.n_input
    assert s.n_output == problem.n_output


def test_concrete_surrogate_train_updates_model_and_size(problem):
    s = _ConcreteSurrogate(problem)
    xs = np.array([[0.1], [0.3], [0.5]])
    ys = np.array([[0.2], [0.4], [0.6]])
    s.train(xs, ys)
    assert s.size == 3
    assert s.model is not None
    assert "xs_mean" in s.model


def test_concrete_surrogate_predict_shape(problem):
    s = _ConcreteSurrogate(problem)
    xs = np.array([[0.2], [0.4]])
    out = s.predict(xs)
    assert out.shape == (2, problem.n_output)


def test_concrete_surrogate_predict_for_adaptive_sampling_zero_variance(problem):
    '''With zero epistemic variance, criteria reduce to mean prediction (LCB/UCB degenerate).'''
    s = _ConcreteSurrogate(problem)
    xs = np.array([[0.1], [0.2]])
    crit = s.predict_for_adaptive_sampling(xs)
    ys = s.predict(xs)
    assert crit.shape == ys.shape
    np.testing.assert_allclose(crit, ys)


def test_concrete_surrogate_full_predict_and_evaluate_performance(problem):
    s = _ConcreteSurrogate(problem)
    xs = np.array([[0.1], [0.2]])
    ys_act = np.array([[0.5], [0.6]])
    fp = s.full_predict(xs)
    assert "ys" in fp
    assert "epistemic_variance" in fp
    assert fp["ys"].shape == (2, problem.n_output)
    perf = s.evaluate_performance(xs, ys_act)
    assert "RMSE" in perf
    assert "MAE" in perf
    assert perf["RMSE"].shape == (problem.n_output,)
    assert perf["MAE"].shape == (problem.n_output,)


@pytest.fixture
def kriging_problem(problem):
    pytest.importorskip("smt")
    return problem


def test_kriging_init_model_list_length(kriging_problem):
    k = Kriging(kriging_problem, model_name="Kriging", train_on_scaled_data=True)
    assert len(k.model) == kriging_problem.n_output
    assert k.size == 0


def test_kriging_train_predict_interpolation(kriging_problem):
    k = Kriging(kriging_problem, train_on_scaled_data=True)
    rng = np.random.default_rng(42)
    xs = rng.uniform(0.05, 0.95, size=(12, kriging_problem.n_input))
    ys = (2.0 * xs).sum(axis=1, keepdims=True)
    k.train(xs, ys)
    assert k.size == 12
    yhat = k.predict(xs)
    np.testing.assert_allclose(yhat, ys, rtol=5e-2, atol=5e-2)


def test_kriging_full_predict_keys_and_variance_shape(kriging_problem):
    k = Kriging(kriging_problem, train_on_scaled_data=True)
    xs = np.linspace(0.1, 0.9, 8).reshape(-1, 1)
    ys = xs**2
    k.train(xs, ys)
    xq = np.array([[0.25], [0.75]])
    out = k.full_predict(xq)
    assert set(out.keys()) >= {"ys", "epistemic_variance"}
    assert out["ys"].shape == (2, kriging_problem.n_output)
    assert out["epistemic_variance"].shape == (2, kriging_problem.n_output)
    assert np.all(out["epistemic_variance"] >= 0)


def test_kriging_evaluate_performance_rmse_mae(kriging_problem):
    k = Kriging(kriging_problem, train_on_scaled_data=True)
    xs = np.linspace(0.1, 0.9, 10).reshape(-1, 1)
    ys = np.sin(np.pi * xs)
    k.train(xs, ys)
    perf = k.evaluate_performance(xs, ys)
    assert "RMSE" in perf and "MAE" in perf
    assert perf["RMSE"].shape == (kriging_problem.n_output,)
    np.testing.assert_allclose(perf["RMSE"], 0.0, atol=1e-2)
    np.testing.assert_allclose(perf["MAE"], 0.0, atol=1e-2)


def test_kriging_predict_for_adaptive_sampling_lcb_minimization(kriging_problem):
    '''Default template output_type is -1 (minimization): LCB = mean - epistemic std.'''
    k = Kriging(kriging_problem, train_on_scaled_data=True)
    xs = np.linspace(0.15, 0.85, 9).reshape(-1, 1)
    ys = xs**2
    k.train(xs, ys)
    xq = np.array([[0.4], [0.6]])
    fp = k.full_predict(xq)
    crit = k.predict_for_adaptive_sampling(xq)
    std = np.sqrt(fp["epistemic_variance"])
    expected = fp["ys"] - std
    np.testing.assert_allclose(crit, expected, rtol=1e-5, atol=1e-8)


def test_kriging_train_on_unscaled_data_predict(kriging_problem):
    k = Kriging(kriging_problem, train_on_scaled_data=False)
    xs = np.linspace(0.2, 0.8, 7).reshape(-1, 1)
    ys = 3.0 * xs
    k.train(xs, ys)
    yhat = k.predict(xs)
    np.testing.assert_allclose(yhat, ys, rtol=0.15, atol=0.15)
