"""
Settings can be built from Python as well as from a JSON file.

The two paths must produce identical objects, and a Python-defined study must
be exportable to a settings file that reads back the same way.
"""

import json
import os

import numpy as np
import pytest

from aeroopt.core import Problem
from aeroopt.core.settings import SettingsData, SettingsProblem
from aeroopt.core.settings_base import REQUIRED, SettingsBase, save_settings
from aeroopt.optimization import (
    OptNSGAII, SettingsDE, SettingsMOEAD, SettingsNRBO, SettingsNSGAII,
    SettingsNSGAIII, SettingsOptimization, SettingsRVEA,
)
from aeroopt.utils import benchmark

DATA_FIELDS = dict(
    name_input=['x1', 'x2'],
    input_low=[0.0, 0.0],
    input_upp=[1.0, 2.0],
    name_output=['cd', 'cl'],
    output_low=[0.0, 0.0],
    output_upp=[1.0, 5.0],
)


@pytest.fixture
def data_settings():
    return SettingsData.from_values('wing_data', **DATA_FIELDS)


class TestBuildFromPython:
    def test_from_values_sets_every_field(self, data_settings):
        assert data_settings.name == 'wing_data'
        assert data_settings.name_input == ['x1', 'x2']
        assert data_settings.n_input == 2
        assert data_settings.n_output == 2
        np.testing.assert_allclose(data_settings.input_upp, [1.0, 2.0])

    def test_settings_mapping_is_equivalent_to_from_values(self, data_settings):
        other = SettingsData('wing_data', settings=dict(DATA_FIELDS))

        assert other.name_input == data_settings.name_input
        np.testing.assert_allclose(other.input_low, data_settings.input_low)
        np.testing.assert_allclose(other.output_upp, data_settings.output_upp)

    def test_precision_defaults_to_continuous(self, data_settings):
        """Omitting precision should mean 'no precision constraint'."""
        np.testing.assert_allclose(data_settings.input_precision, [0.0, 0.0])
        np.testing.assert_allclose(data_settings.output_precision, [0.0, 0.0])

    def test_explicit_precision_is_honoured(self):
        settings = SettingsData.from_values(
            'd', input_precision=[0.1, 0.0], output_precision=[0.0, 0.0],
            **DATA_FIELDS)
        np.testing.assert_allclose(settings.input_precision, [0.1, 0.0])

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match='required key "name_output"'):
            SettingsData.from_values(
                'd', name_input=['x1'], input_low=[0.0], input_upp=[1.0])

    def test_problem_defaults_name_data_settings(self, data_settings):
        problem_settings = SettingsProblem.from_values(
            'wing_problem', data_settings, output_type=[-1, 1])

        assert problem_settings.name_data_settings == data_settings.name
        assert problem_settings.n_objective == 2

    def test_problem_constraints_default_to_empty(self, data_settings):
        problem_settings = SettingsProblem.from_values(
            'p', data_settings, output_type=[-1, 1])
        assert problem_settings.constraint_strings == []
        assert problem_settings.n_constraint == 0

    def test_problem_validates_against_data_settings(self, data_settings):
        with pytest.raises(ValueError, match='Number of output variables'):
            SettingsProblem.from_values('p', data_settings, output_type=[-1])

    def test_list_defaults_are_not_shared_between_instances(self, data_settings):
        first = SettingsProblem.from_values('p1', data_settings, output_type=[-1, 1])
        first.constraint_strings.append('x1 - 1.0')

        second = SettingsProblem.from_values('p2', data_settings, output_type=[-1, 1])

        assert second.constraint_strings == []

    @pytest.mark.parametrize('settings_cls', [
        SettingsOptimization, SettingsNSGAII, SettingsNSGAIII, SettingsRVEA,
        SettingsMOEAD, SettingsDE, SettingsNRBO,
    ])
    def test_optimization_settings_build_from_python(self, settings_cls):
        settings = settings_cls.from_values('demo')
        assert settings.name == 'demo'

    def test_optimization_settings_accept_overrides(self):
        settings = SettingsOptimization.from_values(
            'demo', population_size=16, max_iterations=5, seed=3)

        assert settings.population_size == 16
        assert settings.max_iterations == 5
        assert settings.seed == 3
        # Untouched fields keep their documented defaults.
        assert settings.working_directory == './'
        assert settings.resume is False


class TestRoundTrip:
    def test_python_settings_export_and_reload_identically(self, data_settings, tmp_path):
        problem_settings = SettingsProblem.from_values(
            'wing_problem', data_settings,
            output_type=[-1, 1], constraint_strings=['x1 + x2 - 1.5'])
        opt_settings = SettingsOptimization.from_values(
            'wing_opt', population_size=8, max_iterations=2)

        fname = str(tmp_path / 'settings.json')
        save_settings([data_settings, problem_settings, opt_settings], fname)

        reloaded_data = SettingsData('wing_data', fname_settings=fname)
        reloaded_problem = SettingsProblem('wing_problem', reloaded_data,
                                           fname_settings=fname)
        reloaded_opt = SettingsOptimization('wing_opt', fname_settings=fname)

        assert reloaded_data.name_input == data_settings.name_input
        np.testing.assert_allclose(reloaded_data.input_upp, data_settings.input_upp)
        np.testing.assert_allclose(reloaded_data.output_low, data_settings.output_low)
        assert reloaded_problem.output_type == problem_settings.output_type
        assert reloaded_problem.constraint_strings == problem_settings.constraint_strings
        assert reloaded_opt.population_size == 8
        assert reloaded_opt.max_iterations == 2

    def test_exported_entries_carry_type_and_name(self, data_settings, tmp_path):
        fname = str(tmp_path / 'settings.json')
        save_settings([data_settings], fname)

        with open(fname, encoding='utf-8') as f:
            document = json.load(f)

        entry = next(iter(document.values()))
        assert entry['type'] == 'SettingsData'
        assert entry['name'] == 'wing_data'
        # NumPy arrays must be serialized as plain lists.
        assert isinstance(entry['input_low'], list)

    def test_custom_entry_names(self, data_settings, tmp_path):
        fname = str(tmp_path / 'settings.json')
        save_settings([data_settings], fname, entry_names=['my_key'])

        with open(fname, encoding='utf-8') as f:
            assert list(json.load(f)) == ['my_key']

    def test_entry_names_length_mismatch_raises(self, data_settings, tmp_path):
        with pytest.raises(ValueError, match='entry_names has'):
            save_settings([data_settings], str(tmp_path / 's.json'),
                          entry_names=['a', 'b'])

    def test_to_dict_is_json_serializable(self, data_settings):
        json.dumps(data_settings.to_dict())


class TestPythonDefinedStudy:
    def test_full_optimization_without_any_json_file(self, tmp_path):
        """A study must be runnable with no settings file on disk at all."""
        data_settings = SettingsData.from_values(
            'zdt_data',
            name_input=['x1', 'x2', 'x3'],
            input_low=[0.0] * 3, input_upp=[1.0] * 3,
            name_output=['f1', 'f2'],
            output_low=[-0.1, -1.0], output_upp=[1.1, 10.0])

        problem_settings = SettingsProblem.from_values(
            'zdt_problem', data_settings, output_type=[-1, -1])

        problem = Problem(data_settings, problem_settings)

        opt = OptNSGAII(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'opt', population_size=12, max_iterations=3, seed=7,
                working_directory=str(tmp_path)),
            algorithm_settings=SettingsNSGAII.from_values('alg'),
            user_func=lambda x: (True, benchmark.ZDT1(x)),
            logging=False,
            save_result_files=False,
        )

        opt.main()

        assert not os.path.exists(os.path.join(str(tmp_path), 'settings.json'))
        assert opt.db_total.size > 0
        assert opt.db_elite.size > 0


class TestBackwardCompatibility:
    """The JSON path is unchanged; the template file must still load."""

    @pytest.fixture
    def template_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, 'aeroopt', 'template_settings.json')

    def test_template_still_loads_positionally(self, template_path):
        data_settings = SettingsData('default', template_path)
        problem_settings = SettingsProblem('default', data_settings, template_path)

        assert data_settings.n_input == 1
        assert problem_settings.output_type == [-1]

    def test_missing_file_still_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SettingsData('default', fname_settings='nonexistent.json')

    def test_missing_entry_still_raises_value_error(self, template_path):
        with pytest.raises(ValueError, match='SettingsData .* not found'):
            SettingsData('no_such_entry', fname_settings=template_path)


class TestCustomSettingsClass:
    def test_a_user_defined_settings_class_works_both_ways(self, tmp_path):
        class SettingsMine(SettingsBase):
            _FIELDS = (
                ('alpha', float, 1.0),
                ('label', str, REQUIRED),
            )

        from_python = SettingsMine.from_values('m', alpha=2.5, label='hello')
        assert from_python.alpha == 2.5
        assert from_python.label == 'hello'

        fname = str(tmp_path / 's.json')
        save_settings([from_python], fname)
        from_json = SettingsMine('m', fname_settings=fname)

        assert from_json.alpha == 2.5
        assert from_json.label == 'hello'

    def test_required_field_is_enforced(self):
        class SettingsMine(SettingsBase):
            _FIELDS = (('label', str, REQUIRED),)

        with pytest.raises(ValueError, match='required key "label"'):
            SettingsMine.from_values('m', alpha=1.0)
