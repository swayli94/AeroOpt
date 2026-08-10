'''
Settings of optimization framework and algorithms.

Every settings class reads one entry from a shared JSON file. An entry is
matched by its ``type`` (the class name) and its ``name``:

.. code-block:: json

    {
        "any_entry_key": {
            "type": "SettingsNSGAII",
            "name": "my_algorithm",
            "cross_rate": 0.9
        }
    }

Subclasses of :class:`~aeroopt.core.settings_base.SettingsBase` declare their
fields once in ``_FIELDS``; loading, type conversion and defaulting are handled
by the base class. Every class can be built from a JSON file or directly from
Python, see :mod:`aeroopt.core.settings_base`.
'''

from __future__ import annotations

from typing import Any, Tuple

from aeroopt.core.settings_base import (
    REQUIRED, FieldSpec, SettingsBase, save_settings,
)

__all__ = [
    # Re-exported so a user-defined settings class can be written against
    # `aeroopt.optimization.settings` alone.
    'SettingsBase',
    'FieldSpec',
    'REQUIRED',
    'save_settings',
    'SettingsOptimization',
    'SettingsGeneticOperators',
    'SettingsNSGAII',
    'SettingsNSGAIII',
    'SettingsRVEA',
    'SettingsMOEAD',
    'SettingsDE',
    'SettingsNRBO',
]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


class SettingsOptimization(SettingsBase):
    '''
    Basic settings of the optimization loop, shared by every algorithm.

    Parameters:
    -----------
    name: str
        Name of the optimization settings.
    fname_settings: str
        Name of the settings file. Default is 'settings.json'.

    Attributes:
    -----------
    resume: bool
        Restart from a previously saved database instead of a fresh population.
    population_size: int
        Number of individuals per generation.
    max_iterations: int
        Number of generations after the initial population.
    working_directory: str
        Root directory for ``Calculation``, ``Summary`` and ``Runfiles``.
    info_level_on_screen: int
        Messages with a level at or below this value are echoed to the screen.
    critical_potential_x: float
        Potential at the typical neighbour distance, see
        :class:`~aeroopt.analysis.analyze_database.AnalyzeDatabase`.
    seed: int|None
        Seed of the initial design of experiments.
    force_initial_population_size: int|None
        Overrides ``population_size`` for the initial population only;
        set to 0 to skip sampling entirely (e.g. when resuming).
    '''

    _ALLOW_EXTRA_KEYS = True

    _FIELDS: Tuple[FieldSpec, ...] = (
        ('resume', bool, False),
        ('population_size', int, 64),
        ('max_iterations', int, 100),
        ('fname_db_total', str, 'db-total.json'),
        ('fname_db_elite', str, 'db-elite.json'),
        ('fname_db_resume', str, 'db-resume.json'),
        ('fname_log', str, 'optimization.log'),
        ('working_directory', str, './'),
        ('info_level_on_screen', int, 1),
        ('critical_potential_x', float, 0.2),
        ('seed', _optional_int, None),
        ('force_initial_population_size', _optional_int, None),
    )


class SettingsGeneticOperators(SettingsBase):
    '''
    Settings shared by the SBX / polynomial-mutation based algorithms
    (NSGA-II, NSGA-III, RVEA, MOEA/D).

    Attributes:
    -----------
    cross_rate: float
        Probability of applying simulated binary crossover to a parent pair.
    mut_rate: float
        Mutation rate; the drivers divide it by ``n_input`` so that on average
        one variable per individual is mutated.
    pow_sbx: float
        Distribution index of simulated binary crossover; larger values keep
        children closer to their parents.
    pow_poly: float
        Distribution index of polynomial mutation; larger values give smaller
        mutation steps.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = (
        ('cross_rate', float, 1.0),
        ('mut_rate', float, 1.0),
        ('pow_sbx', float, 20.0),
        ('pow_poly', float, 20.0),
    )


class SettingsNSGAII(SettingsGeneticOperators):
    '''
    Settings of the NSGA-II algorithm.

    Parameters:
    -----------
    name: str
        Name of the NSGAII settings.
    fname_settings: str
        Name of the settings file. Default is 'settings.json'.
    '''


class SettingsNSGAIII(SettingsGeneticOperators):
    '''
    Settings of NSGA-III (same GA operators as NSGA-II, plus reference points).

    Parameters:
    -----------
    name: str
        Name of the NSGA-III settings.
    fname_settings: str
        Name of the settings file. Default is 'settings.json'.

    Attributes:
    -----------
    n_partitions: int|None
        Controls the Das-Dennis reference grid on the (M-1)-simplex.
        If omitted or null, a default is chosen from ``population_size`` when
        running, see
        :meth:`~aeroopt.optimization.moea.DecompositionBasedAlgorithm.suggest_n_partitions`.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = SettingsGeneticOperators._FIELDS + (
        ('n_partitions', _optional_int, None),
    )


class SettingsRVEA(SettingsGeneticOperators):
    '''
    Settings of RVEA (reference-vector guided evolution with APD survival).

    Same GA operators as NSGA-III; the extra parameters follow pymoo's RVEA.

    Attributes:
    -----------
    n_partitions: int|None
        Das-Dennis grid size; if omitted, inferred from ``population_size``.
    alpha: float
        APD penalty exponent applied to the normalized search progress.
    adapt_freq: float
        Fraction of ``max_iterations`` between reference-vector adaptations.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = SettingsGeneticOperators._FIELDS + (
        ('n_partitions', _optional_int, None),
        ('alpha', float, 2.0),
        ('adapt_freq', float, 0.1),
    )


class SettingsMOEAD(SettingsGeneticOperators):
    '''
    Settings of MOEA/D (multiobjective evolutionary algorithm based on decomposition).

    Uses the same SBX/PM operators as NSGA-III. Reference weights are Das-Dennis
    points on the objective simplex; ``population_size`` in
    :class:`SettingsOptimization` must equal the number of those points for the
    chosen ``n_partitions``.

    If the valid archive has fewer feasible individuals than weights after the
    initial evaluation, :class:`~aeroopt.optimization.stochastic.moead.OptMOEAD`
    still initializes by reusing feasible solutions in round-robin order
    (multiple subproblems may share the same individual until neighborhood
    replacement diversifies the slots).

    Attributes:
    -----------
    n_partitions: int|None
        Das-Dennis grid size.
    n_neighbors: int
        Neighbourhood size `T` of each subproblem in weight space.
    prob_neighbor_mating: float
        Probability of drawing both parents from the neighbourhood rather than
        the whole population.
    decomposition: str
        ``'tchebicheff'``, ``'pbi'``, or ``'auto'`` (Tchebycheff for at most two
        objectives, PBI otherwise).
    pbi_theta: float
        PBI penalty weight on the perpendicular distance.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = SettingsGeneticOperators._FIELDS + (
        ('n_partitions', _optional_int, None),
        ('n_neighbors', int, 20),
        ('prob_neighbor_mating', float, 0.9),
        ('decomposition', str, 'auto'),
        ('pbi_theta', float, 5.0),
    )


class SettingsDE(SettingsBase):
    '''
    Settings for differential evolution (DE/rand/1/bin).

    Parameters:
    -----------
    name: str
        Name of the DE settings block in the JSON file.
    fname_settings: str
        Path to the settings file. Default is ``settings.json``.

    Attributes:
    -----------
    scale_factor: float
        Differential weight `F` in `v = x_r0 + F * (x_r1 - x_r2)`.
    cross_rate: float
        Binomial crossover rate `CR`.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = (
        ('scale_factor', float, 0.5),
        ('cross_rate', float, 0.8),
    )


class SettingsNRBO(SettingsBase):
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
    deciding_factor: float
        Probability of applying the Trap Avoidance Operator, which adds a
        perturbation that helps escape local optima.
    '''

    _FIELDS: Tuple[FieldSpec, ...] = (
        ('deciding_factor', float, 0.6),
    )
