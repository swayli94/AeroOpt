'''
Settings of core classes.
'''

from __future__ import annotations

import numpy as np

from typing import Any, Callable, List, Mapping

from aeroopt.core.settings_base import (
    REQUIRED, FieldSpec, SettingsBase, save_settings,
)

__all__ = [
    'SettingsData',
    'SettingsProblem',
    'CustomConstraintFunction',
    'SettingsBase',
    'save_settings',
]


def _float_array(value: Any) -> np.ndarray:
    '''
    Convert a settings value to a 1-D float array.
    '''
    return np.asarray(value, dtype=float).reshape(-1)


def _name_list(value: Any) -> List[str]:
    '''
    Convert a settings value to a list of variable names.
    '''
    return [str(v) for v in value]


class SettingsData(SettingsBase):
    '''
    Settings of the data, i.e., individual in the population.

    The data settings include:

    - name of the data settings
    - input variable's name, lower bound, upper bound and precision
    - output variable's name, lower bound, upper bound and precision
    - critical scaled distance in the input space to distinguish different data

    Parameters:
    -----------
    name: str
        Name of the data settings.
    fname_settings: str
        Path to the settings file. Ignored when `settings` is given.
    settings: Mapping|None
        Field values to use instead of reading a file.

    Example:
    ---------
    >>> # From a settings file
    >>> data_settings = SettingsData('wing_data', fname_settings='settings.json')
    >>>
    >>> # Equivalently, from Python
    >>> data_settings = SettingsData.from_values(
    ...     'wing_data',
    ...     name_input=['x1', 'x2'], input_low=[0.0, 0.0], input_upp=[1.0, 1.0],
    ...     name_output=['cd'], output_low=[0.0], output_upp=[1.0])
    '''

    data_source_dict = {
        'default': 0,
        'previous_database': 1,
        'user_input': 2,
        'DoE': 3,
        'perturbation': 4,
        'evolutionary_operator': 5,
        'surrogate_prediction': 6,
        'space_filling': 7,
        'gradient': 8,
        'sub_direction': 9
    }

    _FIELDS: tuple[FieldSpec, ...] = (
        ('name_input', _name_list, REQUIRED),
        ('input_low', _float_array, REQUIRED),
        ('input_upp', _float_array, REQUIRED),
        # Precision defaults to "continuous" so a Python-defined study only has
        # to state names and bounds.
        ('input_precision', _float_array, None),
        ('name_output', _name_list, REQUIRED),
        ('output_low', _float_array, REQUIRED),
        ('output_upp', _float_array, REQUIRED),
        ('output_precision', _float_array, None),
        ('critical_scaled_distance', float, 1.0e-6),
    )

    def __init__(self, name: str,
            fname_settings: str = 'settings.json',
            *,
            settings: Mapping[str, Any] | None = None):

        super().__init__(name, fname_settings, settings=settings)

        # A missing precision means "no precision constraint" on every variable.
        if self.input_precision is None:
            self.input_precision = np.zeros(len(self.name_input))
        if self.output_precision is None:
            self.output_precision = np.zeros(len(self.name_output))

        self._check_settings()

    @property
    def n_input(self) -> int:
        '''
        Number of input variables.
        '''
        return len(self.name_input)

    @property
    def n_output(self) -> int:
        '''
        Number of output variables.
        '''
        return len(self.name_output)

    @staticmethod
    def apply_precision(variables: np.ndarray, precision: np.ndarray) -> None:
        '''
        Apply the precision to the variables.
        Each variable is rounded to the nearest integer multiple of its precision.
        E.g. precision=2e-3 -> variable must be ..., -0.004, -0.002, 0, 0.002, 0.004, ...

        Parameters
        -------------
        variables: ndarray [n_variables]
            variables to be applied with precision (modified in place)
        precision: ndarray [n_variables]
            precision of the variables (e.g. 2e-3);
            0 means no precision constraint (variable unchanged).

        Returns
        -------------
        None
        '''
        mask_nonzero = precision != 0
        if variables.ndim == 1:
            variables[mask_nonzero] = (
                np.round(variables[mask_nonzero] / precision[mask_nonzero])
                * precision[mask_nonzero]
            )
        else:
            # (n, n_vars): apply precision per column
            variables[:, mask_nonzero] = (
                np.round(variables[:, mask_nonzero] / precision[mask_nonzero])
                * precision[mask_nonzero]
            )

    @staticmethod
    def adjust_bounds(upp: np.ndarray, low: np.ndarray) -> None:
        '''
        Ensure upp >= low, otherwise swap (in place).
        '''
        mask = upp < low
        upp[mask], low[mask] = low[mask].copy(), upp[mask].copy()

    def _check_settings(self) -> None:
        '''
        Check the settings.
        '''
        if self.n_input != len(self.input_low) or self.n_input != len(self.input_upp) or self.n_input != len(self.input_precision):
            raise ValueError('Number of input variables does not match the length of input bounds or precision.')
        if self.n_output != len(self.output_low) or self.n_output != len(self.output_upp) or self.n_output != len(self.output_precision):
            raise ValueError('Number of output variables does not match the length of output bounds or precision.')
        if self.critical_scaled_distance < 0:
            raise ValueError('Critical distance must be non-negative.')

        # Apply the precision to the bounds (in place).
        self.apply_precision(self.input_low, self.input_precision)
        self.apply_precision(self.input_upp, self.input_precision)
        self.apply_precision(self.output_low, self.output_precision)
        self.apply_precision(self.output_upp, self.output_precision)

        # Adjust the bounds (in place).
        self.adjust_bounds(self.input_upp, self.input_low)
        self.adjust_bounds(self.output_upp, self.output_low)

        return None


class CustomConstraintFunction:
    '''
    Template for custom constraint functions.

    Parameters:
    -----------
    data_settings: SettingsData
        Settings of the data.
    '''
    def __init__(self, data_settings: SettingsData):
        self.data_settings = data_settings

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Evaluate the constraint function.
        '''
        raise NotImplementedError('Custom constraint function is not implemented.')

    def _check_settings(self) -> None:
        '''
        Check whether the constraint settings are compatible with the data settings.
        '''


class SettingsProblem(SettingsBase):
    '''
    Settings of the problem for optimization.

    The problem settings include:

    - name of the problem settings
    - output variable's type
    - constraint strings (g(x,y)<=0)
    - constraint functions (g(x,y)<=0)

    Parameters:
    -----------
    name: str
        Name of the problem settings.
    data_settings: SettingsData
        Data settings this problem is checked against.
    fname_settings: str
        Path to the settings file. Ignored when `settings` is given.
    constraint_functions: List[Callable]|None
        Constraint callables `g(x, y) <= 0` that cannot be written as strings.
    settings: Mapping|None
        Field values to use instead of reading a file.

    Example:
    ---------
    >>> # From a settings file
    >>> problem_settings = SettingsProblem('wing', data_settings,
    ...                                    fname_settings='settings.json')
    >>>
    >>> # Equivalently, from Python; `name_data_settings` defaults to the
    >>> # name of the `data_settings` given.
    >>> problem_settings = SettingsProblem.from_values(
    ...     'wing', data_settings,
    ...     output_type=[-1], constraint_strings=['x1 + x2 - 1.0'])
    '''

    output_type_dict = {
        '-1': 'minimum objective',
        '0':  'additional output',
        '1':  'maximum objective',
        '2':  'output for diversity',
    }

    sort_type_dict = {
        0:  'default, by dominance and crowding distance',
        1:  'sorting ID',
        2:  'sorting x',
        3:  'sorting y',
        4:  'sorting objectives',
        5:  'sorting type-2 output',
        6:  'sorting crowding distance and potential',
    }

    dominance_type_dict = {
        0: 'is equal to other',
        1: 'dominates other',
        -1: 'is dominated by other',
        9: 'non-dominated',
    }

    _FIELDS: tuple[FieldSpec, ...] = (
        # Defaults to the name of the `data_settings` passed in, so a
        # Python-defined problem does not have to repeat it.
        ('name_data_settings', str, None),
        ('output_type', lambda v: [int(t) for t in v], REQUIRED),
        ('constraint_strings', lambda v: [str(s) for s in v], []),
    )

    def __init__(self, name: str,
            data_settings: SettingsData,
            fname_settings: str = 'settings.json',
            constraint_functions: List[Callable[[np.ndarray, np.ndarray], float]] | None = None,
            *,
            settings: Mapping[str, Any] | None = None):

        # A fresh list per instance: a shared mutable default would leak
        # constraints appended by one problem into every later problem.
        self.constraint_functions : List[Callable[[np.ndarray, np.ndarray], float]] = (
            [] if constraint_functions is None else list(constraint_functions))

        super().__init__(name, fname_settings, settings=settings)

        if self.name_data_settings is None:
            self.name_data_settings = data_settings.name

        self._check_settings(data_settings)

    @property
    def n_output(self) -> int:
        '''
        Number of output variables.
        '''
        return len(self.output_type)

    @property
    def n_constraint(self) -> int:
        '''
        Number of constraints.
        '''
        return len(self.constraint_strings) + len(self.constraint_functions)

    @property
    def n_objective(self) -> int:
        '''
        Number of objective variables.
        '''
        n_objective = 0
        for out_type in self.output_type:
            if abs(out_type) == 1:
                n_objective += 1
        return n_objective

    def _check_settings(self, data_settings: SettingsData) -> None:
        '''
        Check whether the problem settings are compatible with the data settings.
        '''
        if self.name_data_settings != data_settings.name:
            raise ValueError('Name of data settings does not match.')
        if self.n_output != data_settings.n_output:
            raise ValueError('Number of output variables does not match.')

        # Check whether all the variables in the constraint strings are in the data settings.
        # Only treat token as variable if it looks like an identifier (skip operators like -, +, *, etc.)
        for constraint_str in self.constraint_strings:
            for var in constraint_str.split():
                if not var or not (var[0].isalpha() or var[0] == '_'):
                    continue
                if var not in data_settings.name_input and var not in data_settings.name_output:
                    raise ValueError(f'Variable {var} in constraint string is not in the data settings.')

        return None

