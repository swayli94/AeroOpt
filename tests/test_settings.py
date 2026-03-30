# Tests for aeroopt.core.settings
# 使用 template_settings.json，Windows 路径兼容

import os
import numpy as np
import pytest

from aeroopt.core.settings import SettingsData, SettingsProblem, CustomConstraintFunction


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "aeroopt", "template_settings.json")
    assert os.path.exists(path), f"template_settings.json not found at {path}"
    return path


class TestSettingsData:
    """SettingsData 从 template_settings.json 读取 example_SettingsData."""

    def test_read_settings_data(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        assert sd.name == "default"
        assert sd.name_input == ["x"]
        assert sd.name_output == ["y"]
        assert sd.n_input == 1
        assert sd.n_output == 1

    def test_bounds_and_precision(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        np.testing.assert_array_almost_equal(sd.input_low, [0.0])
        np.testing.assert_array_almost_equal(sd.input_upp, [1.0])
        np.testing.assert_array_almost_equal(sd.input_precision, [1.0e-6])
        np.testing.assert_array_almost_equal(sd.output_low, [-1.0e6])
        np.testing.assert_array_almost_equal(sd.output_upp, [1.0e6])
        assert sd.critical_scaled_distance == 1.0e-6

    def test_settings_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SettingsData("default", fname_settings="nonexistent.json")

    def test_settings_name_not_found(self, settings_path):
        with pytest.raises(ValueError, match="SettingsData .* not found"):
            SettingsData("wrong_name", fname_settings=settings_path)

    def test_apply_precision(self):
        v = np.array([0.0011, 0.0024, 1.0])
        p = np.array([0.001, 0.001, 0.0])
        SettingsData.apply_precision(v, p)
        np.testing.assert_array_almost_equal(v, [0.001, 0.002, 1.0])

    def test_adjust_bounds(self):
        upp = np.array([1.0, 3.0])
        low = np.array([2.0, 1.0])
        SettingsData.adjust_bounds(upp, low)
        np.testing.assert_array_almost_equal(upp, [2.0, 3.0])
        np.testing.assert_array_almost_equal(low, [1.0, 1.0])


class TestSettingsProblem:
    """SettingsProblem 从 template_settings.json 读取 example_SettingsProblem."""

    def test_read_settings_problem(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        sp = SettingsProblem("default", sd, fname_settings=settings_path)
        assert sp.name == "default"
        assert sp.name_data_settings == "default"
        assert sp.output_type == [-1]
        assert sp.constraint_strings == ["y - x"]
        assert sp.n_output == 1
        assert sp.n_constraint == 1
        assert sp.n_objective == 1

    def test_settings_problem_file_not_found(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        with pytest.raises(FileNotFoundError):
            SettingsProblem("default", sd, fname_settings="nonexistent.json")

    def test_settings_problem_name_not_found(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        with pytest.raises(ValueError, match="SettingsProblem .* not found"):
            SettingsProblem("wrong_name", sd, fname_settings=settings_path)

    def test_data_settings_name_mismatch(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        sd.name = "other"
        with pytest.raises(ValueError, match="Name of data settings does not match"):
            SettingsProblem("default", sd, fname_settings=settings_path)


class TestCustomConstraintFunction:
    def test_not_implemented(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        c = CustomConstraintFunction(sd)
        with pytest.raises(NotImplementedError):
            c(np.array([0.5]), np.array([0.5]))
