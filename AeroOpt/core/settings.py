'''
Settings of data, optimization problem and algorithm, etc.
'''

import os
import json
import numpy as np

from typing import List, Callable


class SettingsData(object):
    '''
    Settings of the data, i.e., individual in the population.
    
    The data settings include:
    
    - name of the data settings
    - input variable's name, lower bound, upper bound and precision
    - output variable's name, lower bound, upper bound and precision
    - critical scaled distance in the input space to distinguish different data
    '''
    
    data_source_dict = {
        'default': 0,
        'previous_database': 1,
        'user_input': 2,
        'DoE': 3,
        'random': 4,
        'perturbation': 5,
        'GA': 6,
        'DE': 7,
        'DE-1': 71,
        'DE-2': 72,
        'DE-3': 73,
        'surrogate_prediction': 8,
        'space_filling': 9,
        'gradient': 10,
        'sub_direction': 11
    }
    
    def __init__(self, name: str, 
            fname_settings: str = 'settings.json'):
        
        self.name = name
        
        self.name_input: List[str] = []
        self.input_low: np.ndarray = None
        self.input_upp: np.ndarray = None
        self.input_precision: np.ndarray = None
        
        self.name_output: List[str] = []
        self.output_low: np.ndarray = None
        self.output_upp: np.ndarray = None
        self.output_precision: np.ndarray = None
        
        self.critical_scaled_distance : float = 1.0e-6

        self.read_settings(fname_settings)
        
        self._check_settings()

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')
        
        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_data = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsData' and entry_data['name'] == self.name:
                print(f'>>> SettingsData {self.name} ({entry_name}) read successfully.')
                settings_data = entry_data
        
        if settings_data is None:
            raise ValueError(f'SettingsData {self.name} not found in {fname_settings}.')

        self.name_input = settings_data['name_input']
        self.input_low = np.array(settings_data['input_low'], dtype=float)
        self.input_upp = np.array(settings_data['input_upp'], dtype=float)
        self.input_precision = np.array(settings_data['input_precision'], dtype=float)
        
        self.name_output = settings_data['name_output']
        self.output_low = np.array(settings_data['output_low'], dtype=float)
        self.output_upp = np.array(settings_data['output_upp'], dtype=float)
        self.output_precision = np.array(settings_data['output_precision'], dtype=float)
        
        self.critical_scaled_distance = float(settings_data['critical_scaled_distance'])

        return None
    
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
            raise ValueError(f'Number of input variables does not match the length of input bounds or precision.')
        if self.n_output != len(self.output_low) or self.n_output != len(self.output_upp) or self.n_output != len(self.output_precision):
            raise ValueError(f'Number of output variables does not match the length of output bounds or precision.')
        if self.critical_scaled_distance < 0:
            raise ValueError(f'Critical distance must be non-negative.')
        
        # Apply the precision to the bounds (in place).
        self.apply_precision(self.input_low, self.input_precision)
        self.apply_precision(self.input_upp, self.input_precision)
        self.apply_precision(self.output_low, self.output_precision)
        self.apply_precision(self.output_upp, self.output_precision)
        
        # Adjust the bounds (in place).
        self.adjust_bounds(self.input_upp, self.input_low)
        self.adjust_bounds(self.output_upp, self.output_low)
        
        return None
    

class CustomConstraintFunction(object):
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
        raise NotImplementedError(f'Custom constraint function is not implemented.')
    
    def _check_settings(self) -> None:
        '''
        Check whether the constraint settings are compatible with the data settings.
        '''
    
    
class SettingsProblem(object):
    '''
    Settings of the problem for optimization.
    
    The problem settings include:
    
    - name of the problem settings
    - output variable's type
    - constraint strings (g(x,y)<=0)
    - constraint functions (g(x,y)<=0)
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
    }
    
    dominance_type_dict = {
        0: 'is equal to other',
        1: 'dominates other',
        -1: 'is dominated by other',
        9: 'non-dominated',
    }
    
    def __init__(self, name: str, 
            data_settings: SettingsData,
            fname_settings: str = 'settings.json'):
        
        self.name = name
        
        self.name_data_settings : str = 'default'
        self.output_type : List[int] = [-1]
        self.constraint_strings : List[str] = []
        self.constraint_functions : List[Callable[[np.ndarray, np.ndarray], float]] = []
        
        self.read_settings(fname_settings)
        
        self._check_settings(data_settings)
    
    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')
        
        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        settings_optimization = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsProblem' and entry_data['name'] == self.name:
                print(f'>>> SettingsProblem {self.name} ({entry_name}) read successfully.')
                settings_optimization = entry_data

        if settings_optimization is None:
            raise ValueError(f'SettingsProblem {self.name} not found in {fname_settings}.')
        
        self.name_data_settings = settings_optimization['name_data_settings']
        self.output_type = list(map(int, settings_optimization['output_type']))
        self.constraint_strings = list(settings_optimization['constraint_strings'])
        
        return None

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
            raise ValueError(f'Name of data settings does not match.')
        if self.n_output != data_settings.n_output:
            raise ValueError(f'Number of output variables does not match.')
        
        # Check whether all the variables in the constraint strings are in the data settings.
        # Only treat token as variable if it looks like an identifier (skip operators like -, +, *, etc.)
        for constraint_str in self.constraint_strings:
            for var in constraint_str.split():
                if not var or not (var[0].isalpha() or var[0] == '_'):
                    continue
                if var not in data_settings.name_input and var not in data_settings.name_output:
                    raise ValueError(f'Variable {var} in constraint string is not in the data settings.')
        
        return None



    
    