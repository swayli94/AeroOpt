'''
Settings of optimization framework and algorithms.
'''

import os
import json
from typing import Dict, Any

# JSON entry metadata; not applied as optimization fields.
_SETTINGS_OPTIMIZATION_METADATA_KEYS = frozenset({'type', 'name'})


class SettingsOptimization(object):
    '''
    Basic settings of optimization.
    
    Parameters:
    -----------
    name: str
        Name of the optimization settings.
    fname_settings: str
        Name of the settings file.
        Default is 'settings.json'.
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):
        
        self.name = name
        
        self.resume : bool = False
        self.population_size : int = 64
        self.max_iterations : int = 100
        self.fname_db_total : str = 'db-total.json'
        self.fname_db_elite : str = 'db-elite.json'
        self.fname_db_population : str = 'db-population.json'
        self.fname_db_resume : str = 'db-resume.json'
        self.fname_log : str = 'optimization.log'
        self.working_directory : str = './'
        self.info_level_on_screen : int = 1
        self.critical_potential_x : float = 0.2
        self.seed : int|None = None
        self.force_initial_population_size : int|None = None
        
        self.settings: Dict[str, Any] = {}
        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')
        
        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        settings_opt= None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsOptimization' and entry_data['name'] == self.name:
                print(f'>>> SettingsOptimization {self.name} ({entry_name}) read successfully.')
                settings_opt = entry_data

        if settings_opt is None:
            raise ValueError(f'SettingsOptimization {self.name} not found in {fname_settings}.')

        self.settings = settings_opt
        for key, value in settings_opt.items():
            if key in _SETTINGS_OPTIMIZATION_METADATA_KEYS:
                continue
            if key == 'resume':
                self.resume = bool(value)
            elif key in ('population_size', 'max_iterations', 'info_level_on_screen'):
                setattr(self, key, int(value))
            elif key == 'critical_potential_x':
                self.critical_potential_x = float(value)
            elif key.startswith('fname_') or key in ('working_directory', 'fname_log'):
                setattr(self, key, str(value))
            elif key == 'seed':
                self.seed = int(value) if value is not None else None
            elif key == 'force_initial_population_size':
                self.force_initial_population_size = int(value) if value is not None else None
            else:
                setattr(self, key, value)

        return None


class SettingsNSGAII(object):
    '''
    Settings of NSGAII algorithm.

    Parameters:
    -----------
    name: str
        Name of the NSGAII settings.
    fname_settings: str
        Name of the settings file.
        Default is 'settings.json'.
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name

        self.cross_rate: float = 1.0
        self.mut_rate: float = 1.0
        self.pow_sbx: float = 20.0
        self.pow_poly: float = 20.0
        self.reserve_ratio: float = 0.3

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_nsgaii = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsNSGAII' and entry_data['name'] == self.name:
                print(f'>>> SettingsNSGAII {self.name} ({entry_name}) read successfully.')
                settings_nsgaii = entry_data

        if settings_nsgaii is None:
            raise ValueError(f'SettingsNSGAII {self.name} not found in {fname_settings}.')

        self.cross_rate = float(settings_nsgaii['cross_rate'])
        self.mut_rate = float(settings_nsgaii['mut_rate'])
        self.pow_sbx = float(settings_nsgaii['pow_sbx'])
        self.pow_poly = float(settings_nsgaii['pow_poly'])
        self.reserve_ratio = float(settings_nsgaii['reserve_ratio'])
        
        return None


class SettingsDE(object):
    '''
    Settings for differential evolution (DE/rand/1/bin).

    Parameters:
    -----------
    name: str
        Name of the DE settings block in the JSON file.
    fname_settings: str
        Path to the settings file. Default is ``settings.json``.
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name
        self.scale_factor: float = 0.5
        self.cross_rate: float = 0.8

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_de = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsDE' and entry_data['name'] == self.name:
                print(f'>>> SettingsDE {self.name} ({entry_name}) read successfully.')
                settings_de = entry_data

        if settings_de is None:
            raise ValueError(f'SettingsDE {self.name} not found in {fname_settings}.')

        self.scale_factor = float(settings_de['scale_factor'])
        self.cross_rate = float(settings_de['cross_rate'])

        return None


class SettingsNRBO(object):
    '''
    Settings of NRBO (Newton-Raphson-based optimizer).

    Parameters:
    -----------
    name: str
        Name of the NRBO settings block in the JSON file.
    fname_settings: str
        Path to the settings file. Default is 'settings.json'.
        
    Attributes:
    -----------
    name: str
        Name of the NRBO settings block in the JSON file.
    deciding_factor: float
        Probability of applying the TAO operator,
        introducing an additional perturbation to avoid getting trapped in a local optimum.
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name
        self.deciding_factor: float = 0.6

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_entry = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsNRBO' and entry_data['name'] == self.name:
                print(f'>>> SettingsNRBO {self.name} ({entry_name}) read successfully.')
                settings_entry = entry_data

        if settings_entry is None:
            raise ValueError(f'SettingsNRBO {self.name} not found in {fname_settings}.')

        self.deciding_factor = float(settings_entry.get('deciding_factor', 0.6))
        return None


class SettingsNSGAIII(object):
    '''
    Settings of NSGA-III algorithm (same GA operators as NSGA-II, plus reference points).

    Parameters:
    -----------
    name: str
        Name of the NSGA-III settings.
    fname_settings: str
        Name of the settings file.
        Default is 'settings.json'.

    Notes:
    ------
    `n_partitions` controls the Das-Dennis reference grid on the (M-1)-simplex.
    If omitted or null, a default is chosen from `population_size` when running
    (see `DecompositionBasedAlgorithm.suggest_n_partitions`).
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name

        self.cross_rate: float = 1.0
        self.mut_rate: float = 1.0
        self.pow_sbx: float = 20.0
        self.pow_poly: float = 20.0
        self.reserve_ratio: float = 0.3
        self.n_partitions: int|None = None

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_entry = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsNSGAIII' and entry_data['name'] == self.name:
                print(f'>>> SettingsNSGAIII {self.name} ({entry_name}) read successfully.')
                settings_entry = entry_data

        if settings_entry is None:
            raise ValueError(f'SettingsNSGAIII {self.name} not found in {fname_settings}.')

        self.cross_rate = float(settings_entry['cross_rate'])
        self.mut_rate = float(settings_entry['mut_rate'])
        self.pow_sbx = float(settings_entry['pow_sbx'])
        self.pow_poly = float(settings_entry['pow_poly'])
        self.reserve_ratio = float(settings_entry['reserve_ratio'])
        npart = settings_entry.get('n_partitions', None)
        self.n_partitions = int(npart) if npart is not None else None

        return None


class SettingsRVEA(object):
    '''
    Settings of RVEA (reference-vector guided evolution with APD survival).

    Same GA operators as NSGA-III; extra parameters follow pymoo RVEA:
    `alpha` (APD penalty) and `adapt_freq` (reference-vector adaptation).

    `n_partitions` selects the Das-Dennis grid; if omitted, a default is
    inferred from `population_size` (see `DecompositionBasedAlgorithm.suggest_n_partitions`).
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name

        self.cross_rate: float = 1.0
        self.mut_rate: float = 1.0
        self.pow_sbx: float = 20.0
        self.pow_poly: float = 20.0
        self.reserve_ratio: float = 0.3
        self.n_partitions: int|None = None
        self.alpha: float = 2.0
        self.adapt_freq: float = 0.1

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_entry = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsRVEA' and entry_data['name'] == self.name:
                print(f'>>> SettingsRVEA {self.name} ({entry_name}) read successfully.')
                settings_entry = entry_data

        if settings_entry is None:
            raise ValueError(f'SettingsRVEA {self.name} not found in {fname_settings}.')

        self.cross_rate = float(settings_entry['cross_rate'])
        self.mut_rate = float(settings_entry['mut_rate'])
        self.pow_sbx = float(settings_entry['pow_sbx'])
        self.pow_poly = float(settings_entry['pow_poly'])
        self.reserve_ratio = float(settings_entry['reserve_ratio'])
        npart = settings_entry.get('n_partitions', None)
        self.n_partitions = int(npart) if npart is not None else None
        self.alpha = float(settings_entry.get('alpha', 2.0))
        self.adapt_freq = float(settings_entry.get('adapt_freq', 0.1))

        return None


class SettingsMOEAD(object):
    '''
    Settings of MOEA/D (multiobjective evolutionary algorithm based on decomposition).

    Uses the same SBX/PM operators as NSGA-III. Reference weights are Das–Dennis
    points on the objective simplex; ``population_size`` in ``SettingsOptimization``
    must equal the number of those points for the chosen ``n_partitions``
    (see ``DecompositionBasedAlgorithm.suggest_n_partitions`` / combinatorial count).

    If the valid archive has fewer feasible individuals than weights after the
    initial evaluation, ``OptMOEAD`` still initializes by reusing feasible
    solutions in round-robin order (multiple subproblems may share the same
    individual until neighborhood replacement diversifies the slots).

    Parameters:
    -----------
    name: str
        Name of the MOEA/D settings block in the JSON file.
    fname_settings: str
        Path to the settings file. Default is ``settings.json``.
    '''
    def __init__(self, name: str,
            fname_settings: str = 'settings.json'):

        self.name = name

        self.cross_rate: float = 1.0
        self.mut_rate: float = 1.0
        self.pow_sbx: float = 20.0
        self.pow_poly: float = 20.0
        self.n_partitions: int|None = None
        self.n_neighbors: int = 20
        self.prob_neighbor_mating: float = 0.9
        self.decomposition: str = 'auto'
        self.pbi_theta: float = 5.0

        self.read_settings(fname_settings)

    def read_settings(self, fname_settings: str) -> None:
        '''
        Read settings from json file.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings_entry = None
        for entry_name, entry_data in settings.items():
            if entry_data['type'] == 'SettingsMOEAD' and entry_data['name'] == self.name:
                print(f'>>> SettingsMOEAD {self.name} ({entry_name}) read successfully.')
                settings_entry = entry_data

        if settings_entry is None:
            raise ValueError(f'SettingsMOEAD {self.name} not found in {fname_settings}.')

        self.cross_rate = float(settings_entry['cross_rate'])
        self.mut_rate = float(settings_entry['mut_rate'])
        self.pow_sbx = float(settings_entry['pow_sbx'])
        self.pow_poly = float(settings_entry['pow_poly'])
        npart = settings_entry.get('n_partitions', None)
        self.n_partitions = int(npart) if npart is not None else None
        self.n_neighbors = int(settings_entry.get('n_neighbors', 20))
        self.prob_neighbor_mating = float(
            settings_entry.get('prob_neighbor_mating', 0.9))
        self.decomposition = str(settings_entry.get('decomposition', 'auto'))
        self.pbi_theta = float(settings_entry.get('pbi_theta', 5.0))

        return None

