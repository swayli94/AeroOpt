# Tests for AeroOpt.core.individual
# 使用 template_settings.json 构建 Problem，Windows 路径兼容

import os
import numpy as np
import pytest

from AeroOpt.core.settings import SettingsData, SettingsProblem
from AeroOpt.core.problem import Problem
from AeroOpt.core.individual import Individual


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "template_settings.json")
    assert os.path.exists(path), f"template_settings.json not found at {path}"
    return path


@pytest.fixture(scope="module")
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    p = Problem(sd, sp)
    # Individual.__init__ / __str__ 依赖 problem.name
    p.name = sp.name
    # Individual.objectives 依赖 problem.output_type
    p.output_type = sp.output_type
    return p


@pytest.fixture
def ind(problem):
    x = np.array([0.5])
    y = np.array([0.5])
    return Individual(problem, x, ID=1, y=y)


class TestIndividualBasics:
    def test_init(self, ind, problem):
        np.testing.assert_array_almost_equal(ind.x, [0.5])
        np.testing.assert_array_almost_equal(ind.y, [0.5])
        assert ind.ID == 1
        assert ind.problem is problem
        assert ind.generation == 0
        assert ind.source == "default"

    def test_repr_str(self, ind):
        assert "1" in repr(ind)
        assert "Individual" in str(ind) and "1" in str(ind)

    def test_y_scalar_converted(self, problem):
        ind = Individual(problem, np.array([0.5]), ID=2, y=3.0)
        np.testing.assert_array_almost_equal(ind.y, [3.0])

    def test_source2int(self, ind):
        assert ind.source2int == 0
        ind.source = "DE"
        assert ind.source2int == 7

    def test_int2source(self):
        assert Individual.int2source(0) == "default"
        assert Individual.int2source(7) == "DE"


class TestIndividualObjectives:
    def test_objectives(self, ind):
        obj = ind.objectives
        assert obj.shape == (1,)
        np.testing.assert_array_almost_equal(obj, [0.5])


class TestIndividualConstraints:
    def test_eval_constraints(self, ind):
        sum_v, violations = ind.eval_constraints()
        # constraint y - x <= 0, x=0.5 y=0.5 => 0
        np.testing.assert_almost_equal(sum_v, 0.0)
        np.testing.assert_array_almost_equal(violations, [0.0])
        ind2 = Individual(ind.problem, np.array([0.8]), ID=3, y=np.array([0.9]))
        sum_v2, _ = ind2.eval_constraints()
        np.testing.assert_almost_equal(sum_v2, 0.1)


class TestIndividualDominance:
    def test_check_dominance_same_problem(self, problem):
        a = Individual(problem, np.array([0.2]), ID=1, y=np.array([0.1]))
        b = Individual(problem, np.array([0.5]), ID=2, y=np.array([0.3]))
        a.eval_constraints()
        b.eval_constraints()
        # minimize: a better
        assert a.check_dominance(b) == 1
        assert b.check_dominance(a) == -1

    def test_check_dominance_invalid_other_type(self, ind):
        with pytest.raises(ValueError, match="Must compare individuals"):
            ind.check_dominance(1)

    def test_check_dominance_equal_y(self, problem):
        a = Individual(problem, np.array([0.5]), ID=1, y=np.array([0.5]))
        b = Individual(problem, np.array([0.5]), ID=2, y=np.array([0.5]))
        a.eval_constraints()
        b.eval_constraints()
        assert a.check_dominance(b) == 0
        assert b.check_dominance(a) == 0


class TestIndividualLt:
    def test_lt_by_id(self, ind, problem):
        other = Individual(problem, np.array([0.5]), ID=2, y=np.array([0.5]))
        ind.sort_type = 1
        other.sort_type = 1
        assert (ind < other) is True
        assert (other < ind) is False

    def test_lt_by_dominance(self, problem):
        a = Individual(problem, np.array([0.2]), ID=1, y=np.array([0.2]))
        b = Individual(problem, np.array([0.5]), ID=2, y=np.array([0.5]))
        a.pareto_rank = 0
        b.pareto_rank = 0
        a.crowding_distance = 2.0
        b.crowding_distance = 1.0
        a.eval_constraints()
        b.eval_constraints()
        assert (a < b) is True
        assert (b < a) is False

    def test_lt_not_individual(self, ind):
        with pytest.raises(TypeError):
            _ = ind < 1
