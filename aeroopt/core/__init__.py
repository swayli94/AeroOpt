'''
Core functions and classes.
'''
from aeroopt.core.problem import Problem, StaleCaseFolderError
from aeroopt.core.individual import (
    Individual,
    ID_UNASSIGNED,
    SORT_BY_DOMINANCE_AND_CROWDING,
    SORT_BY_ID,
    SORT_BY_X,
    SORT_BY_Y,
    SORT_BY_OBJECTIVES,
    SORT_BY_DIVERSITY_OUTPUT,
    SORT_BY_CROWDING,
)
from aeroopt.core.database import Database
from aeroopt.core.settings import (
    SettingsData,
    SettingsProblem,
    CustomConstraintFunction,
)
from aeroopt.core.settings_base import (
    SettingsBase,
    FieldSpec,
    REQUIRED,
    save_settings,
)
from aeroopt.core.utils import (
    init_log, log, check_folder, compare_ndarray,
)
from aeroopt.core.mp_evaluation import MultiProcessEvaluation

__all__ = [
    'SettingsData',
    'SettingsProblem',
    'CustomConstraintFunction',
    'SettingsBase',
    'FieldSpec',
    'REQUIRED',
    'save_settings',
    'Problem',
    'StaleCaseFolderError',
    'Individual',
    'Database',
    'MultiProcessEvaluation',
    'init_log',
    'log',
    'check_folder',
    'compare_ndarray',

    'ID_UNASSIGNED',
    'SORT_BY_DOMINANCE_AND_CROWDING',
    'SORT_BY_ID',
    'SORT_BY_X',
    'SORT_BY_Y',
    'SORT_BY_OBJECTIVES',
    'SORT_BY_DIVERSITY_OUTPUT',
    'SORT_BY_CROWDING',
]
