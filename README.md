# AeroOpt (`aeroopt`)

A multi-objective (and single-objective) optimization framework for engineering
workflows where **evaluating a design is the expensive part** — a CFD run, an FEA
solve, or any external executable that takes minutes to hours per sample.

That premise drives the design:

- **Every evaluation is kept.** Designs accumulate in a persistent `Database`
  that can be written to JSON/Excel, reloaded, merged, sub-setted and restarted
  from. The optimizer works on an archive, not a transient generation.
- **Evaluation is pluggable and parallel.** Objectives can be Python callables
  or external solvers driven in their own working folders;
  `MultiProcessEvaluation` spreads a generation across processes.
- **The loop is open.** `PreProcess` / `PostProcess` hooks let you repair,
  screen or replace candidates before an expensive evaluation is spent on them.
- **Failure is expected.** Diverged solves and constraint violations are
  recorded rather than dropped, and the search adapts to how much feasible data
  actually exists.

## Features

- **Problems and data**: `Problem`, `Individual`, `Database`; constraint strings
  and custom constraint callables; JSON / Excel serialization.
- **Evaluation**: built-in Python objectives or external executables;
  `MultiProcessEvaluation` for parallel runs (Linux and Windows).
- **Optimization loop**: `OptBaseFramework` with pluggable `PreProcess` /
  `PostProcess`.
- **Evolutionary algorithms**: NSGA-II, NSGA-III, RVEA, MOEA/D, differential
  evolution (MODE-style) and NRBO.
- **Surrogates and hybrids**: `SAO` and `SBO` in `aeroopt.optimization.hybrid`;
  `aeroopt.utils.surrogate` defines the surrogate interface (Kriging via SMT).
- **Analysis**: `AnalyzeDatabase` for input-space crowding metrics, potential
  fields and clustering; standard test functions in `aeroopt.utils.benchmark`.

For richer visualization and decision-making around a computed Pareto set, see
[pymoo](https://pymoo.org/).

## Requirements

- Python **≥ 3.9**
- `numpy`, `scipy`, `scikit-learn`, `numexpr`, `pydoe>=0.9.8`, `openpyxl`

## Installation

```bash
pip install aeroopt                 # core
pip install "aeroopt[surrogate]"    # + smt, for Kriging
pip install "aeroopt[examples]"     # + matplotlib, for the example scripts
```

Editable install from a clone:

```bash
pip install -e ".[surrogate,examples,tests]"
```

## Quick start

```python
import numpy as np
from aeroopt.core import Problem, SettingsData, SettingsProblem
from aeroopt.optimization import OptNSGAII, SettingsOptimization, SettingsNSGAII

def evaluate(x: np.ndarray):
    """Return (succeed, y); succeed=False records a failed evaluation."""
    return True, np.array([x[0], 1.0 - np.sqrt(x[0]) + x[1]])

data_settings = SettingsData.from_values(
    'demo_data',
    name_input=['x1', 'x2'], input_low=[0.0, 0.0], input_upp=[1.0, 1.0],
    name_output=['f1', 'f2'], output_low=[-0.1, -1.0], output_upp=[1.1, 10.0],
)

problem_settings = SettingsProblem.from_values(
    'demo_problem', data_settings,
    output_type=[-1, -1],                        # both minimized
    constraint_strings=['x1 ** 2 + x2 ** 2 - 0.64'],
)

opt = OptNSGAII(
    problem=Problem(data_settings, problem_settings),
    optimization_settings=SettingsOptimization.from_values(
        'demo_opt', population_size=32, max_iterations=20, seed=42),
    algorithm_settings=SettingsNSGAII.from_values('demo_alg'),
    user_func=evaluate,
)
opt.main()

print(opt.db_elite.size, 'non-dominated designs')
```

### Configuration: Python or JSON

The settings above are defined inline. The same objects can equivalently be read
from a JSON file, which is the better choice for a study worth
version-controlling:

```python
data_settings = SettingsData('demo_data', fname_settings='settings.json')
```

`save_settings([...], 'settings.json')` converts a Python-defined study into
that file, and it reads back unchanged. See
`aeroopt/template_settings.json` for a template with every supported entry.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

The documentation covers the settings reference, the architecture, and the
principles behind each algorithm (dominance and crowding, SBX/polynomial
mutation, DE, NRBO, NSGA-III niching, RVEA's angle-penalized distance, MOEA/D
decomposition, and surrogate-driven optimization).

## Package layout

| Package | Role |
| --------- | ------ |
| `aeroopt.core` | `Problem`, `Individual`, `Database`, settings (JSON or Python), `MultiProcessEvaluation`, logging |
| `aeroopt.sampling` | Design-of-experiments samplers on the unit hypercube |
| `aeroopt.optimization` | `OptBaseFramework`, `PreProcess` / `PostProcess`, settings, operators |
| `aeroopt.optimization.stochastic` | NSGA-II/III, RVEA, MOEA/D, DE, NRBO |
| `aeroopt.optimization.hybrid` | `SAO`, `SBO` and their post-processing |
| `aeroopt.analysis` | `AnalyzeDatabase`: statistics, crowding metrics, clustering |
| `aeroopt.utils` | `benchmark`, `surrogate` |

## Algorithms

| Algorithm | Driver | Selection principle |
| ----------- | -------- | --------------------- |
| NSGA-II | `OptNSGAII` | Non-dominated rank, then crowding distance |
| NSGA-III | `OptNSGAIII` | Non-dominated rank, then reference-point niching |
| RVEA | `OptRVEA` | Angle-penalized distance to adaptive reference vectors |
| MOEA/D | `OptMOEAD` | Scalarized subproblems with neighbourhood replacement |
| DE | `OptDE` | DE/rand/1/bin offspring on a rank-and-crowding archive |
| NRBO | `OptNRBO` | Newton-Raphson search rule (single objective) |
| SBO | `SBO` | All candidates from an optimization run on a surrogate |
| SAO | `SAO` | Evolutionary and surrogate candidates mixed per iteration |

## Examples (`example/`)

Scripts prepend the repository root to `sys.path` so they run from a clone
without installing; remove that block if you installed the package.

| Folder | Script | Summary |
| -------- | -------- | --------- |
| `1-database-io` | `example_core_functions.py` | Problem and database setup; JSON / Excel I/O |
| `2-mp-evaluation` | `example_mpEvaluation.py` | Parallel evaluation, built-in and external |
| `3-database-evaluation` | `example_database_evaluation.py` | Serial vs. multiprocessing vs. external |
| `4-pre-process` | `example_pre_process.py` | Custom `PreProcess` and candidate repair |
| `5-evolutionary-algorithm` | `example_dominance_based_algorithm.py` | Dominance, crowding and selection tools |
| `5-evolutionary-algorithm` | `example_pareto_analysis.py` | Lagging reference directions on a front |
| `6-single-objective-optimization` | `example_soo.py` | NSGA-II vs. DE vs. NRBO |
| `7-multi-objective-optimization` | `example_nsgaii.py`, ... | ZDT suite across all MOEAs |
| `8-surrogate-hybrid-optimization` | `example_kriging.py`, `example_sbo.py`, `example_sao.py` | Kriging and the hybrid frameworks |

## Tests

```bash
pytest
```

## Contributing

Development principles, architecture notes and a change checklist are in
[AGENTS.md](AGENTS.md) — written for humans and AI agents alike.

An agent skill for *using* the library lives in [skills/aeroopt/](skills/aeroopt/):
point your assistant at it, or read `skills/aeroopt/SKILL.md` yourself for a
condensed guide to the settings, the algorithms and the common pitfalls.

## Repository

<https://github.com/swayli94/AeroOpt>
