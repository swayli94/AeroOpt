# Tests for AeroOpt.core.problem
# 使用 template_settings.json，Windows 路径兼容

import os
import tempfile
import numpy as np
import pytest

from AeroOpt.core.settings import SettingsData, SettingsProblem
from AeroOpt.core.problem import Problem


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "AeroOpt", "template_settings.json")
    assert os.path.exists(path), f"template_settings.json not found at {path}"
    return path


@pytest.fixture(scope="module")
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


class TestProblemBasics:
    def test_n_input_n_output(self, problem):
        assert problem.n_input == 1
        assert problem.n_output == 1
        assert problem.n_constraint == 1
        assert problem.n_objective == 1

    def test_name_inputs_outputs(self, problem):
        # core 中 Problem 无 name_inputs/name_outputs，使用 data_settings 的 name_input/name_output
        assert problem.data_settings.name_input == ["x"]
        assert problem.data_settings.name_output == ["y"]

    def test_eq(self, problem):
        sd = problem.data_settings
        sp = problem.problem_settings
        p2 = Problem(sd, sp)
        assert problem == p2
        sp2 = SettingsProblem("default", sd, fname_settings=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AeroOpt", "template_settings.json"))
        p3 = Problem(sd, sp2)
        assert problem == p3


class TestProblemBoundsAndScale:
    def test_check_bounds_x(self, problem):
        assert problem.check_bounds_x(np.array([0.5])) == True
        assert problem.check_bounds_x(np.array([-0.1])) == False
        assert problem.check_bounds_x(np.array([1.5])) == False

    def test_check_bounds_y(self, problem):
        assert problem.check_bounds_y(np.array([0.0])) == True
        assert problem.check_bounds_y(np.array([-0.5e6])) == True  # 在 [output_low, output_upp] 内
        assert problem.check_bounds_y(np.array([2.0e6])) == False

    def test_apply_bounds_x(self, problem):
        x = np.array([1.5])
        out = problem.apply_bounds_x(x)
        np.testing.assert_array_almost_equal(x, [1.0])
        assert out is False
        x2 = np.array([0.5])
        assert problem.apply_bounds_x(x2) is True

    def test_apply_bounds_y(self, problem):
        y = np.array([2.0e6])
        problem.apply_bounds_y(y)
        np.testing.assert_array_almost_equal(y, [1.0e6])

    def test_scale_x(self, problem):
        x = np.array([0.5])
        s = problem.scale_x(x, reverse=False)
        np.testing.assert_array_almost_equal(s, [0.5])
        s2 = problem.scale_x(s, reverse=True)
        np.testing.assert_array_almost_equal(s2, [0.5])

    def test_scale_y(self, problem):
        y = np.array([0.0])
        s = problem.scale_y(y, reverse=False)
        np.testing.assert_array_almost_equal(s, [0.5])
        s2 = problem.scale_y(s, reverse=True)
        np.testing.assert_array_almost_equal(s2, [0.0])


class TestProblemConstraints:
    def test_eval_constraint_string(self, problem):
        # constraint: y - x <= 0  => violation = y - x (if > 0)
        v = problem.eval_constraint_string("y - x", np.array([0.2]), np.array([0.1]))
        np.testing.assert_almost_equal(v, -0.1)
        v2 = problem.eval_constraint_string("y - x", np.array([0.1]), np.array([0.2]))
        np.testing.assert_almost_equal(v2, 0.1)

    def test_eval_constraints(self, problem):
        sum_v, violations = problem.eval_constraints(np.array([0.1]), np.array([0.2]))
        np.testing.assert_array_almost_equal(violations, [0.1])
        np.testing.assert_almost_equal(sum_v, 0.1)
        sum_v2, _ = problem.eval_constraints(np.array([0.5]), np.array([0.3]))
        np.testing.assert_almost_equal(sum_v2, 0.0)


class TestProblemPareto:
    def test_check_pareto_dominance(self, problem):
        # output_type [-1] => minimize
        # y1 better than y2 if y1 < y2
        r = problem.check_pareto_dominance(np.array([1.0]), np.array([2.0]))
        assert r == 1
        r = problem.check_pareto_dominance(np.array([2.0]), np.array([1.0]))
        assert r == -1
        r = problem.check_pareto_dominance(np.array([1.0]), np.array([1.0]))
        assert r == 0


class TestProblemGetOutputByType:
    def test_get_output_by_type(self, problem):
        y = np.array([1.0])
        out = problem.get_output_by_type(y, [1, -1])
        np.testing.assert_array_almost_equal(out, [1.0])


class TestProblemPerturb:
    def test_perturb_scaled_x(self, problem):
        np.random.seed(42)
        scaled = np.array([0.5])
        out = problem.perturb_scaled_x(scaled, n_perturb=3, dx=0.01)
        assert out.shape == (3, 1)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_perturb_x(self, problem):
        np.random.seed(42)
        x = np.array([0.5])
        out = problem.perturb_x(x, n_perturb=2, dx=0.01)
        assert out.shape == (2, 1)
        assert problem.check_bounds_x(out[0]) and problem.check_bounds_x(out[1])


class TestProblemDistance:
    def test_calculate_scaled_distance(self, problem):
        x1 = np.array([0.0])
        x2 = np.array([1.0])
        d = problem.calculate_scaled_distance(x1, x2, is_scaled_x=True)
        assert d.size == 1
        np.testing.assert_almost_equal(d.flat[0], 1.0)
        x1b = np.array([[0.0], [0.5]])
        x2b = np.array([[1.0]])
        db = problem.calculate_scaled_distance(x1b, x2b, is_scaled_x=True)
        assert db.shape == (2, 1)


class TestProblemIO:
    def test_write_input(self, problem):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            fname = f.name
        try:
            problem.write_input(fname, np.array([0.5]))
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
            assert "x" in content and "0.5" in content
        finally:
            os.remove(fname)

    def test_read_output_no_file(self, problem):
        succeed, y = problem.read_output("nonexistent_file.txt")
        assert succeed is False
        assert y.shape == (1,)

    def test_read_output(self, problem):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("y  0.25\n")
            fname = f.name
        try:
            succeed, y = problem.read_output(fname)
            assert succeed is True
            np.testing.assert_array_almost_equal(y, [0.25])
        finally:
            os.remove(fname)

    def test_read_input_no_file(self, problem):
        succeed, x = problem.read_input("nonexistent_file.txt")
        assert succeed is False
        assert x.shape == (1,)

    def test_read_input(self, problem):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x  0.75\n")
            fname = f.name
        try:
            succeed, x = problem.read_input(fname)
            assert succeed is True
            np.testing.assert_array_almost_equal(x, [0.75])
        finally:
            os.remove(fname)
