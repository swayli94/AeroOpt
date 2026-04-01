# Tests for aeroopt.core.mpEvaluation
# 使用内置 func 或 template_usr_func，不依赖外部 run.bat

import numpy as np
import pytest

from aeroopt.core.mpEvaluation import MultiProcessEvaluation, template_usr_func


class TestTemplateUsrFunc:
    def test_template_usr_func(self):
        x = np.array([1.0, 2.0, 3.0])
        succeed, y = template_usr_func(x)
        assert succeed is True
        np.testing.assert_array_almost_equal(y, [1 + 4 + 9])


class TestMultiProcessEvaluationWithFunc:
    def test_serial_evaluate(self):
        def my_func(x, **kwargs):
            return True, np.array([np.sum(x)])

        mp = MultiProcessEvaluation(2, 1, func=my_func, n_process=None)
        xs = np.array([[1.0, 2.0], [3.0, 4.0]])
        list_succeed, ys = mp.evaluate(xs)
        assert list_succeed == [True, True]
        np.testing.assert_array_almost_equal(ys, [[3.0], [7.0]])

    def test_serial_with_template_usr_func(self):
        mp = MultiProcessEvaluation(3, 1, func=template_usr_func, n_process=None)
        xs = np.array([[1.0, 0.0, 0.0]])
        list_succeed, ys = mp.evaluate(xs)
        assert list_succeed == [True]
        np.testing.assert_array_almost_equal(ys, [[1.0]])


class TestMultiProcessEvaluationInit:
    def test_init_with_func(self):
        mp = MultiProcessEvaluation(1, 1, func=lambda x: (True, np.array([0.0])))
        assert mp.dim_input == 1
        assert mp.dim_output == 1
        assert mp.func is not None
        assert mp.n_process is None

    def test_init_without_func_evaluate_requires_external_args(self):
        # __init__ 允许 func=None；无 func 时 evaluate 必须提供 list_name 与 prob
        mp = MultiProcessEvaluation(1, 1, func=None)
        xs = np.array([[1.0]])
        # New validation order checks `prob` first, so message differs from older list_name-first behavior.
        with pytest.raises(Exception, match="Problem object `prob`"):
            mp.evaluate(xs)
