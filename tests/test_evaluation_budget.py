"""Evaluation budget and segmented continuation, driven through the real loop.

Both features exist for the same kind of study: an evaluation costs minutes to
hours, the comparison between two workflows has to be over an equal solver
cost rather than an equal generation count, and the run is long enough that it
is completed in segments.

These tests drive `OptDE.main()` rather than the methods in isolation, because
what matters is the interaction: the budget has to survive the boundary
between two segments, and the generation history has to survive it with the
budget applied.
"""

import json
import os

import numpy as np
import pytest

from aeroopt.core import Problem, SettingsData, SettingsProblem
from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization
from aeroopt.utils import benchmark

N_INPUT = 3
N_OUTPUT = 2
POPULATION_SIZE = 6


def _write_settings(path, work_dir):
    config = {
        'data': {
            'type': 'SettingsData', 'name': 'd',
            'name_input': [f'x{i + 1}' for i in range(N_INPUT)],
            'input_low': [0.0] * N_INPUT,
            'input_upp': [1.0] * N_INPUT,
            'input_precision': [0.0] * N_INPUT,
            'name_output': [f'y{i + 1}' for i in range(N_OUTPUT)],
            'output_low': [-1.0] * N_OUTPUT,
            'output_upp': [10.0] * N_OUTPUT,
            'output_precision': [0.0] * N_OUTPUT,
            'critical_scaled_distance': 1.0e-8,
        },
        'problem': {
            'type': 'SettingsProblem', 'name': 'p', 'name_data_settings': 'd',
            'output_type': [-1] * N_OUTPUT, 'constraint_strings': [],
        },
        'opt': {
            'type': 'SettingsOptimization', 'name': 'o', 'resume': False,
            'population_size': POPULATION_SIZE, 'max_iterations': 2,
            'working_directory': work_dir, 'info_level_on_screen': 0, 'seed': 7,
        },
        'de': {
            'type': 'SettingsDE', 'name': 'a',
            'scale_factor': 0.5, 'cross_rate': 0.9,
        },
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f)


def _objective(x):
    return True, benchmark.ZDT1(x)


@pytest.fixture
def settings_file(tmp_path):
    path = os.path.join(str(tmp_path), 'settings.json')
    _write_settings(path, str(tmp_path))
    return path


def _problem(settings_file):
    data_settings = SettingsData('d', fname_settings=settings_file)
    problem_settings = SettingsProblem('p', data_settings,
                                       fname_settings=settings_file)
    return Problem(data_settings, problem_settings)


def _driver(settings_file, seed=3, **overrides):
    optimization_settings = SettingsOptimization('o', fname_settings=settings_file)
    for key, value in overrides.items():
        setattr(optimization_settings, key, value)

    return OptDE(
        problem=_problem(settings_file),
        optimization_settings=optimization_settings,
        algorithm_settings=SettingsDE('a', fname_settings=settings_file),
        user_func=_objective,
        logging=False,
        save_result_files=False,
        rng=np.random.default_rng(seed),
    )


def test_budget_stops_the_run_before_the_generation_limit(settings_file):
    '''A budget of 13 with a population of 6 must buy 13 evaluations, not 18.'''
    opt = _driver(settings_file, max_iterations=100, max_evaluations=13)

    opt.main()

    assert opt.new_evaluations == 13
    assert opt.remaining_evaluations == 0
    # Three passes through the loop: 6 + 6 + a final batch trimmed to 1.
    assert opt.iteration == 3
    assert opt.iteration < opt.max_iterations
    assert opt.db_total.size == POPULATION_SIZE + 13


def test_budget_excludes_the_initial_population(settings_file):
    '''The DoE has its own size control; the budget is what the loop spends.'''
    opt = _driver(settings_file, max_iterations=100, max_evaluations=6)

    opt.main()

    assert opt.new_evaluations == 6
    assert opt.db_total.size == POPULATION_SIZE + 6


def test_no_budget_keeps_generation_based_stopping(settings_file):
    '''`max_evaluations = 0` must leave the existing behaviour untouched.'''
    opt = _driver(settings_file, max_iterations=2, max_evaluations=0)

    opt.main()

    assert opt.iteration == 2
    assert opt.db_total.size == POPULATION_SIZE * 3


def _write_resume_file(opt, settings_file):
    summary = os.path.join(os.path.dirname(settings_file), 'Summary')
    os.makedirs(summary, exist_ok=True)
    path = os.path.join(summary, opt.optimization_settings.fname_db_resume)
    opt.db_total.output_database_json(path)
    return path


def test_a_continuation_keeps_generations_ids_and_its_own_budget(settings_file):
    '''The wingbox continuation in miniature: segment two resumes segment one.

    Without `resume_preserve_generation` the second segment would restart the
    generation count at zero, and anything reading "the previous generation"
    across the boundary --- a convergence history, an operator inspecting the
    last batch --- would see a run that never happened.
    '''
    first = _driver(settings_file, max_iterations=100, max_evaluations=12)
    first.main()

    assert first.new_evaluations == 12
    first_generations = [int(indi.generation) for indi in first.db_total.individuals]
    assert max(first_generations) == 2
    last_id = max(int(indi.ID) for indi in first.db_total.individuals)

    _write_resume_file(first, settings_file)

    second = _driver(
        settings_file, seed=5, max_iterations=100, max_evaluations=12,
        resume=True, resume_preserve_generation=True,
        force_initial_population_size=0,
    )
    second.main()

    # The resumed stock is not charged to the second segment's budget.
    assert second.new_evaluations == 12
    assert second.db_total.size == first.db_total.size + 12

    resumed = [indi for indi in second.db_total.individuals
               if indi.source == 'previous_database']
    fresh = [indi for indi in second.db_total.individuals
             if indi.source != 'previous_database']

    assert len(resumed) == first.db_total.size
    assert sorted(int(indi.generation) for indi in resumed) == sorted(first_generations)

    # New offspring continue the history instead of colliding with it.
    assert min(int(indi.generation) for indi in fresh) == 3
    assert min(int(indi.ID) for indi in fresh) == last_id + 1


def test_a_continuation_without_the_flag_restarts_the_generation_count(settings_file):
    '''The default is still a fresh study seeded by an existing database.'''
    first = _driver(settings_file, max_iterations=100, max_evaluations=12)
    first.main()
    _write_resume_file(first, settings_file)

    second = _driver(
        settings_file, seed=5, max_iterations=100, max_evaluations=6,
        resume=True, force_initial_population_size=0,
    )
    second.main()

    resumed = [indi for indi in second.db_total.individuals
               if indi.source == 'previous_database']

    assert all(int(indi.generation) == 0 for indi in resumed)


def test_a_budget_smaller_than_the_doe_does_not_cut_the_doe(settings_file):
    '''The initial population is the starting stock, not a charge on the budget.

    The trim runs inside `evaluate_db_candidate`, which the initial population
    goes through as well, so a study whose budget is smaller than its DoE
    would have had that DoE silently cut down to the budget.
    '''
    opt = _driver(settings_file, max_iterations=100, max_evaluations=2)

    opt.main()

    assert opt.db_total.size == POPULATION_SIZE + 2
    assert opt.new_evaluations == 2


def test_the_budget_is_not_spent_on_candidates_dropped_as_duplicates(settings_file):
    '''Trimming happens after the duplicate screen, not before it.

    A candidate dropped because the study already evaluated it must not
    consume a slot the budget would have paid for; otherwise a run ends short
    of its budget with nothing to say why.
    '''
    opt = _driver(settings_file, max_iterations=100, max_evaluations=18)

    opt.main()

    assert opt.new_evaluations == 18
    assert opt.remaining_evaluations == 0
