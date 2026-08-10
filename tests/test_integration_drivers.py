"""
End-to-end smoke tests: every optimization driver must complete a short run.

The unit tests exercise the algorithms' static operators in isolation. These
tests run the real loop --- sampling, evaluation, archive updates, elite
selection --- so a driver that no longer wires up to `OptBaseFramework` is
caught even when its operators are still correct.
"""

import json
import os

import numpy as np
import pytest

from aeroopt.core import Problem, SettingsData, SettingsProblem
from aeroopt.optimization import (
    OptDE,
    OptMOEAD,
    OptNRBO,
    OptNSGAII,
    OptNSGAIII,
    OptRVEA,
    SettingsDE,
    SettingsMOEAD,
    SettingsNRBO,
    SettingsNSGAII,
    SettingsNSGAIII,
    SettingsOptimization,
    SettingsRVEA,
)
from aeroopt.utils import benchmark

N_INPUT = 3

# MOEA/D requires population_size == number of Das-Dennis points.
# For 2 objectives and n_partitions=5 that is 6.
POPULATION_SIZE = 6
N_PARTITIONS = 5


def _write_settings(path: str, work_dir: str, n_output: int) -> None:
    config = {
        'data': {
            'type': 'SettingsData', 'name': 'd',
            'name_input': [f'x{i + 1}' for i in range(N_INPUT)],
            'input_low': [0.0] * N_INPUT,
            'input_upp': [1.0] * N_INPUT,
            'input_precision': [0.0] * N_INPUT,
            'name_output': [f'y{i + 1}' for i in range(n_output)],
            'output_low': [-1.0] * n_output,
            'output_upp': [10.0] * n_output,
            'output_precision': [0.0] * n_output,
            'critical_scaled_distance': 1.0e-8,
        },
        'problem': {
            'type': 'SettingsProblem', 'name': 'p', 'name_data_settings': 'd',
            'output_type': [-1] * n_output, 'constraint_strings': [],
        },
        'opt': {
            'type': 'SettingsOptimization', 'name': 'o', 'resume': False,
            'population_size': POPULATION_SIZE, 'max_iterations': 2,
            'working_directory': work_dir, 'info_level_on_screen': 9, 'seed': 7,
        },
        'nsgaii': {
            'type': 'SettingsNSGAII', 'name': 'a',
            'cross_rate': 0.9, 'mut_rate': 0.9, 'pow_sbx': 20.0, 'pow_poly': 20.0,
        },
        'nsgaiii': {
            'type': 'SettingsNSGAIII', 'name': 'a',
            'cross_rate': 0.9, 'mut_rate': 0.9, 'pow_sbx': 20.0, 'pow_poly': 20.0,
            'n_partitions': None,
        },
        'rvea': {
            'type': 'SettingsRVEA', 'name': 'a',
            'cross_rate': 0.9, 'mut_rate': 0.9, 'pow_sbx': 20.0, 'pow_poly': 20.0,
            'n_partitions': None, 'alpha': 2.0, 'adapt_freq': 0.5,
        },
        'moead': {
            'type': 'SettingsMOEAD', 'name': 'a',
            'cross_rate': 0.9, 'mut_rate': 0.9, 'pow_sbx': 20.0, 'pow_poly': 20.0,
            'n_partitions': N_PARTITIONS, 'n_neighbors': 3,
            'prob_neighbor_mating': 0.9, 'decomposition': 'auto', 'pbi_theta': 5.0,
        },
        'de': {
            'type': 'SettingsDE', 'name': 'a',
            'scale_factor': 0.5, 'cross_rate': 0.9,
        },
        'nrbo': {
            'type': 'SettingsNRBO', 'name': 'a', 'deciding_factor': 0.6,
        },
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f)


def _build_problem(tmp_path, n_output: int):
    path = os.path.join(str(tmp_path), 'settings.json')
    _write_settings(path, str(tmp_path), n_output)
    data_settings = SettingsData('d', fname_settings=path)
    problem_settings = SettingsProblem('p', data_settings, fname_settings=path)
    return Problem(data_settings, problem_settings), path


def _multi_objective(x: np.ndarray):
    return True, benchmark.ZDT1(x)


def _single_objective(x: np.ndarray):
    return True, np.array([float(np.sum(x ** 2))])


DRIVERS = [
    ('nsgaii', OptNSGAII, SettingsNSGAII, 2, _multi_objective),
    ('nsgaiii', OptNSGAIII, SettingsNSGAIII, 2, _multi_objective),
    ('rvea', OptRVEA, SettingsRVEA, 2, _multi_objective),
    ('moead', OptMOEAD, SettingsMOEAD, 2, _multi_objective),
    ('de', OptDE, SettingsDE, 2, _multi_objective),
    ('nrbo', OptNRBO, SettingsNRBO, 1, _single_objective),
]


@pytest.mark.parametrize('name,driver,settings_cls,n_output,func', DRIVERS,
                         ids=[d[0] for d in DRIVERS])
def test_driver_completes_a_short_run(tmp_path, name, driver, settings_cls,
                                      n_output, func):
    problem, path = _build_problem(tmp_path, n_output)

    opt = driver(
        problem=problem,
        optimization_settings=SettingsOptimization('o', fname_settings=path),
        algorithm_settings=settings_cls('a', fname_settings=path),
        user_func=func,
        logging=False,
        save_result_files=False,
        rng=np.random.default_rng(3),
    )

    opt.main()

    assert opt.iteration == 2
    assert opt.db_total.size > 0
    assert opt.db_valid.size > 0
    assert opt.db_elite.size > 0
    # The elite set is the first non-dominated front of the valid archive.
    assert opt.db_elite.size <= opt.db_valid.size


@pytest.mark.parametrize('name,driver,settings_cls,n_output,func', DRIVERS,
                         ids=[d[0] for d in DRIVERS])
def test_driver_accepts_an_injected_generator(tmp_path, name, driver,
                                              settings_cls, n_output, func):
    """Every driver must expose `rng`; OptRVEA and OptMOEAD used not to."""
    problem, path = _build_problem(tmp_path, n_output)

    opt = driver(
        problem=problem,
        optimization_settings=SettingsOptimization('o', fname_settings=path),
        algorithm_settings=settings_cls('a', fname_settings=path),
        user_func=func,
        logging=False,
        save_result_files=False,
        rng=np.random.default_rng(11),
    )

    assert isinstance(opt.rng, np.random.Generator)


def _flaky_objective(n_output: int):
    """An objective that fails on part of the design space, like a diverging solver."""
    def func(x: np.ndarray):
        if x[0] > 0.6:
            return False, np.zeros(n_output)
        if n_output == 1:
            return True, np.array([float(np.sum(x ** 2))])
        return True, benchmark.ZDT1(x)
    return func


@pytest.mark.parametrize('name,driver,settings_cls,n_output,_func', DRIVERS,
                         ids=[d[0] for d in DRIVERS])
def test_driver_survives_failed_evaluations(tmp_path, name, driver, settings_cls,
                                            n_output, _func):
    """
    A failed evaluation leaves an individual with an empty output vector in
    ``db_total``. Breeding from that archive used to raise a broadcast error, so
    a single diverging solver run took the whole optimization down.
    """
    problem, path = _build_problem(tmp_path, n_output)

    opt = driver(
        problem=problem,
        optimization_settings=SettingsOptimization('o', fname_settings=path),
        algorithm_settings=settings_cls('a', fname_settings=path),
        user_func=_flaky_objective(n_output),
        logging=False,
        save_result_files=False,
        rng=np.random.default_rng(5),
    )

    opt.main()

    n_failed = sum(1 for indi in opt.db_total.individuals
                   if not indi.valid_evaluation)

    assert n_failed > 0, 'the objective did not actually fail; test is vacuous'
    # Failures are remembered in the total archive but excluded from the valid one.
    assert opt.db_valid.size == opt.db_total.size - n_failed
    assert opt.db_elite.size > 0
    assert all(indi.valid_evaluation for indi in opt.db_valid.individuals)


def test_seed_in_settings_makes_a_run_reproducible(tmp_path):
    """`seed` used to cover only the initial sample, not the operators."""
    problem, path = _build_problem(tmp_path, 2)

    runs = []
    for _ in range(2):
        opt = OptNSGAII(
            problem=problem,
            optimization_settings=SettingsOptimization('o', fname_settings=path),
            algorithm_settings=SettingsNSGAII('a', fname_settings=path),
            user_func=_multi_objective,
            logging=False,
            save_result_files=False,
        )
        opt.main()
        runs.append(opt.db_total.get_xs())

    assert runs[0].shape == runs[1].shape
    np.testing.assert_allclose(runs[0], runs[1])
