#!/usr/bin/env python3
"""
Scaffold a runnable aeroopt study: settings.json + run.py.

Writing settings.json by hand is the most error-prone part of using aeroopt
(entries are matched by `type` + `name`, constraint strings must be
space-separated, MOEA/D constrains the population size). This generates a
consistent pair and verifies that the settings actually load.

Examples
--------
    python new_study.py --dir my_study
    python new_study.py --dir wing --n-input 6 --objectives 2 --algorithm rvea
    python new_study.py --dir shape --algorithm moead --population 32
    python new_study.py --dir solver --external          # external executable
    python new_study.py --dir cheap --objectives 1 --algorithm nrbo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Algorithm name -> (settings type, driver class, settings class)
ALGORITHMS = {
    'nsgaii': ('SettingsNSGAII', 'OptNSGAII', 'SettingsNSGAII'),
    'nsgaiii': ('SettingsNSGAIII', 'OptNSGAIII', 'SettingsNSGAIII'),
    'rvea': ('SettingsRVEA', 'OptRVEA', 'SettingsRVEA'),
    'moead': ('SettingsMOEAD', 'OptMOEAD', 'SettingsMOEAD'),
    'de': ('SettingsDE', 'OptDE', 'SettingsDE'),
    'nrbo': ('SettingsNRBO', 'OptNRBO', 'SettingsNRBO'),
}

SINGLE_OBJECTIVE_ONLY = {'nrbo'}
NEEDS_TWO_OBJECTIVES = {'moead'}


def n_das_dennis_points(n_objective: int, n_partitions: int) -> int:
    """Number of Das-Dennis reference points; MOEA/D needs population_size == this."""
    return math.comb(n_partitions + n_objective - 1, n_objective - 1)


def moead_partitions_for(n_objective: int, population_size: int) -> int:
    """Smallest partition count whose reference-point total reaches population_size."""
    p = 1
    while n_das_dennis_points(n_objective, p) < population_size:
        p += 1
    return p


def build_algorithm_entry(algorithm: str, name: str, n_objective: int,
                          population_size: int) -> dict:
    entry_type = ALGORITHMS[algorithm][0]
    entry = {'type': entry_type, 'name': name}

    if algorithm in ('nsgaii', 'nsgaiii', 'rvea', 'moead'):
        entry.update({'cross_rate': 0.9, 'mut_rate': 0.9,
                      'pow_sbx': 20.0, 'pow_poly': 20.0})

    if algorithm in ('nsgaiii', 'rvea'):
        entry['n_partitions'] = None
    if algorithm == 'rvea':
        entry.update({'alpha': 2.0, 'adapt_freq': 0.1})
    if algorithm == 'moead':
        entry.update({
            'n_partitions': moead_partitions_for(n_objective, population_size),
            'n_neighbors': min(20, population_size),
            'prob_neighbor_mating': 0.9,
            'decomposition': 'auto',
            'pbi_theta': 5.0,
        })
    if algorithm == 'de':
        entry.update({'scale_factor': 0.5, 'cross_rate': 0.9})
    if algorithm == 'nrbo':
        entry['deciding_factor'] = 0.6

    return entry


def build_settings(args) -> dict:
    n_in, n_out = args.n_input, args.objectives
    prefix = args.prefix

    input_names = [f'x{i + 1}' for i in range(n_in)]
    output_names = [f'y{i + 1}' for i in range(n_out)]

    # A constraint is included only when there are at least two inputs, so the
    # generated example is always valid. Note the space-separated tokens.
    constraints = ([f'{input_names[0]} ** 2 + {input_names[1]} ** 2 - 0.64']
                   if n_in >= 2 and args.with_constraint else [])

    settings = {
        'data': {
            'type': 'SettingsData',
            'name': f'{prefix}_data',
            'name_input': input_names,
            'input_low': [0.0] * n_in,
            'input_upp': [1.0] * n_in,
            # 0.0 means continuous; set e.g. 0.1 to snap a variable to a grid.
            'input_precision': [0.0] * n_in,
            'name_output': output_names,
            # Scaling bounds, NOT constraints. Set to each output's plausible range.
            'output_low': [-1.0] * n_out,
            'output_upp': [10.0] * n_out,
            'output_precision': [0.0] * n_out,
            'critical_scaled_distance': 1.0e-8,
        },
        'problem': {
            'type': 'SettingsProblem',
            'name': f'{prefix}_problem',
            'name_data_settings': f'{prefix}_data',
            # -1 minimize, 1 maximize, 0 record only, 2 diversity
            'output_type': [-1] * n_out,
            # g(x, y) <= 0; tokens MUST be space-separated
            'constraint_strings': constraints,
        },
        'opt': {
            'type': 'SettingsOptimization',
            'name': f'{prefix}_opt',
            'resume': False,
            'population_size': args.population,
            'max_iterations': args.iterations,
            'fname_db_total': 'db-total.json',
            'fname_db_elite': 'db-elite.json',
            'fname_db_resume': 'db-resume.json',
            'fname_log': 'optimization.log',
            'working_directory': './',
            'info_level_on_screen': 1,
            'critical_potential_x': 0.2,
            'seed': args.seed,
            'force_initial_population_size': None,
        },
        'alg': build_algorithm_entry(args.algorithm, f'{prefix}_alg',
                                     n_out, args.population),
    }

    return settings


RUN_TEMPLATE = '''"""
{title}

Generated by the aeroopt skill scaffolder. Edit `evaluate` to call your solver.
"""

import numpy as np

from aeroopt.core import Problem, SettingsData, SettingsProblem
from aeroopt.optimization import {driver}, SettingsOptimization, {settings_cls}

FNAME = 'settings.json'
PREFIX = '{prefix}'


def evaluate(x: np.ndarray):
    """
    Evaluate one design.

    Returns
    -------
    succeed: bool
        False records a failed evaluation. It stays in `db_total` as evidence
        that the region is troublesome, but is excluded from `db_valid`.
    y: np.ndarray [{n_out}]
        Output vector, in the order of `name_output` in settings.json.
    """
{evaluate_body}


def main() -> None:
    data_settings = SettingsData(f'{{PREFIX}}_data', fname_settings=FNAME)
    problem_settings = SettingsProblem(
        f'{{PREFIX}}_problem', data_settings, fname_settings=FNAME)
    problem = Problem(data_settings, problem_settings)

    opt = {driver}(
        problem=problem,
        optimization_settings=SettingsOptimization(
            f'{{PREFIX}}_opt', fname_settings=FNAME),
        algorithm_settings={settings_cls}(f'{{PREFIX}}_alg', fname_settings=FNAME),
        user_func={user_func},
{mp_arg}    )

    opt.main()

    print(f'total  : {{opt.db_total.size}}')
    print(f'valid  : {{opt.db_valid.size}}')
    print(f'elite  : {{opt.db_elite.size}}')

    # Non-dominated feasible designs.
    xs = opt.db_elite.get_xs()
    ys = opt.db_elite.get_ys()
    for i in range(opt.db_elite.size):
        print(f'  x={{np.round(xs[i], 4)}}  y={{np.round(ys[i], 4)}}')

    opt.db_total.json_to_excel('Summary/db-total.json', 'Summary/db-total.xlsx')


if __name__ == '__main__':
    main()
'''

PYTHON_EVAL_BODY = '''    # TODO: replace with the real evaluation.
    y = np.array([{expr}])
    return True, y'''

EXTERNAL_EVAL_NOTE = '''# NOTE: user_func is None, so aeroopt drives an external program.
# For each design it creates Calculation/<ID>/, copies in the CONTENTS of
# Runfiles/, writes input.txt, and runs run.sh (run.bat on Windows) there.
#
#   Runfiles/run.sh        input.txt          output.txt
#   -------------------    ---------------    ---------------
#   python solver.py         x1  0.35           y1  0.1225
#                            x2  0.80           y2  2.4400
#
# The solver only has to read input.txt and write output.txt in that
# `name value` format. A missing file or variable is a failed evaluation.
'''


def build_run_script(args) -> str:
    _, driver, settings_cls = ALGORITHMS[args.algorithm]
    n_out = args.objectives

    if args.external:
        evaluate_body = ('    raise NotImplementedError(\n'
                         '        "user_func is unused: this study runs an external solver.")')
        user_func = 'None'
        title = EXTERNAL_EVAL_NOTE
    else:
        # A trivial separable placeholder so the generated study runs as-is.
        terms = [f'float(np.sum(x ** 2))'] if n_out == 1 else (
            ['float(x[0])'] + [f'float(np.sum(x[1:]) + {i})' for i in range(1, n_out)])
        evaluate_body = PYTHON_EVAL_BODY.format(expr=', '.join(terms))
        user_func = 'evaluate'
        title = f'aeroopt study: {args.algorithm.upper()} on a {n_out}-objective problem.'

    mp_arg = ''
    if args.parallel:
        mp_arg = ('        mp_evaluation=MultiProcessEvaluation(\n'
                  '            dim_input=problem.n_input, dim_output=problem.n_output,\n'
                  f'            func={user_func}, n_process={args.parallel}),\n')

    script = RUN_TEMPLATE.format(
        title=title, driver=driver, settings_cls=settings_cls,
        prefix=args.prefix, n_out=n_out,
        evaluate_body=evaluate_body, user_func=user_func, mp_arg=mp_arg)

    if args.parallel:
        script = script.replace(
            'from aeroopt.core import Problem, SettingsData, SettingsProblem',
            'from aeroopt.core import (\n'
            '    MultiProcessEvaluation, Problem, SettingsData, SettingsProblem)')

    return script


def _ensure_aeroopt_importable() -> None:
    """
    Allow running from a clone where aeroopt was never pip-installed.

    This file lives at <repo>/skills/aeroopt/scripts/new_study.py, so the
    repository root is three levels up. Appending (not prepending) keeps a real
    installed aeroopt winning over the checkout.
    """
    try:
        import aeroopt  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    if os.path.isdir(os.path.join(repo_root, 'aeroopt')):
        sys.path.append(repo_root)


def validate(directory: str, prefix: str) -> bool:
    """Load the generated settings to prove they are consistent."""
    _ensure_aeroopt_importable()

    try:
        from aeroopt.core import Problem, SettingsData, SettingsProblem
    except ImportError as e:
        print(f'  ! aeroopt not importable, skipping validation: {e}')
        return True

    fname = os.path.join(directory, 'settings.json')
    cwd = os.getcwd()
    try:
        data_settings = SettingsData(f'{prefix}_data', fname_settings=fname)
        problem_settings = SettingsProblem(
            f'{prefix}_problem', data_settings, fname_settings=fname)
        problem = Problem(data_settings, problem_settings)
        print(f'  validated: n_input={problem.n_input} '
              f'n_output={problem.n_output} '
              f'n_objective={problem.n_objective} '
              f'n_constraint={problem.n_constraint}')
        return True
    except Exception as e:
        print(f'  ! validation FAILED: {type(e).__name__}: {e}')
        return False
    finally:
        os.chdir(cwd)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True,
                   help='Directory to create the study in.')
    p.add_argument('--prefix', default='study',
                   help='Prefix for the settings entry names (default: study).')
    p.add_argument('--n-input', type=int, default=3,
                   help='Number of design variables (default: 3).')
    p.add_argument('--objectives', type=int, default=2,
                   help='Number of objectives (default: 2).')
    p.add_argument('--algorithm', default='nsgaii', choices=sorted(ALGORITHMS),
                   help='Optimization algorithm (default: nsgaii).')
    p.add_argument('--population', type=int, default=32,
                   help='Population size (default: 32).')
    p.add_argument('--iterations', type=int, default=20,
                   help='Number of generations (default: 20).')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed; makes the run reproducible (default: 42).')
    p.add_argument('--external', action='store_true',
                   help='Drive an external executable instead of a Python function.')
    p.add_argument('--parallel', type=int, metavar='N', default=0,
                   help='Evaluate with N worker processes.')
    p.add_argument('--with-constraint', action='store_true', default=True,
                   help='Include an example constraint (default: on).')
    p.add_argument('--no-constraint', dest='with_constraint', action='store_false',
                   help='Omit the example constraint.')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.algorithm in SINGLE_OBJECTIVE_ONLY and args.objectives != 1:
        print(f'error: {args.algorithm} supports only single-objective problems; '
              f'got --objectives {args.objectives}', file=sys.stderr)
        return 2

    if args.algorithm in NEEDS_TWO_OBJECTIVES and args.objectives < 2:
        print(f'error: {args.algorithm} requires at least two objectives',
              file=sys.stderr)
        return 2

    if args.algorithm == 'moead':
        p = moead_partitions_for(args.objectives, args.population)
        exact = n_das_dennis_points(args.objectives, p)
        if exact != args.population:
            print(f'note: MOEA/D needs population_size == the number of '
                  f'Das-Dennis points; adjusting {args.population} -> {exact} '
                  f'(n_partitions={p}).')
            args.population = exact

    os.makedirs(args.dir, exist_ok=True)

    settings_path = os.path.join(args.dir, 'settings.json')
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(build_settings(args), f, indent=4, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {settings_path}')

    run_path = os.path.join(args.dir, 'run.py')
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(build_run_script(args))
    print(f'wrote {run_path}')

    if args.external:
        runfiles = os.path.join(args.dir, 'Runfiles')
        os.makedirs(runfiles, exist_ok=True)
        script = os.path.join(runfiles, 'run.sh')
        with open(script, 'w', encoding='utf-8') as f:
            f.write('#!/bin/sh\n# Called with the case folder as the working directory.\n'
                    '# Read input.txt, write output.txt.\n'
                    'python solver.py\n')
        os.chmod(script, 0o755)
        print(f'wrote {script}  (add your solver alongside it)')

    ok = validate(args.dir, args.prefix)

    print()
    print(f'next: cd {args.dir} && python run.py')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
