# AeroOpt (`aeroopt`)

A multi-objective (and single-objective) optimization framework for engineering workflows.

This framework is developed to handle the complex needs of:

- calling external tools for evaluation
- parallel evaluation
- pre- and post-processing of the population
- user manipulation of the population

This framework is also developed for other purposes, such as:

- hybridization of different optimization algorithms, machine learning tools, etc.
- adaptive sampling

## Features

- **Problems and data**: `Problem`, `Individual`, `Database`; constraint strings and custom constraint callables; databases can be serialized to JSON / Excel.
- **Evaluation**: built-in Python objectives or external executables; `MultiProcessEvaluation` for parallel runs (examples cover Linux and Windows).
- **Optimization loop**: `OptBaseFramework` with pluggable `PreProcess` / `PostProcess`; examples show population pre/post-processing and user hooks.
- **Evolutionary algorithms**: NSGA-II, NSGA-III, RVEA, MOEA/D, differential evolution (MODE-style), NRBO, and more; `DominanceBasedAlgorithm` supplies non-dominated sorting, crowding, parent selection, and related helpers.
- **Surrogates and hybrids**: **SAO** and **SBO** in `aeroopt.optimization.hybrid`; `aeroopt.utils.surrogate` defines surrogate interfaces (e.g. Kriging with SMT or similar backends).
- **Analysis and utilities**: `AnalyzeDatabase` in `aeroopt.analysis.analyze_database`; standard test functions in `aeroopt.utils.benchmark` (e.g. Rastrigin, ZDT suites).

For richer visualization and decision-making around multi-objective optimization, see [pymoo](https://pymoo.org/index.html).

## Requirements

- Python **≥ 3.9**
- Core dependencies: `numpy`, `scipy`, `scikit-learn` (see `pyproject.toml`)

## Installation

Repository: <https://github.com/swayli94/AeroOpt>

From PyPI:

```bash
pip install aeroopt
```

Editable install from a clone of this repository:

```bash
pip install -e .
```

## Package layout

| Package | Role |
|---------|------|
| `aeroopt.core` | `Problem`, `Individual`, `Database`, settings types, `MultiProcessEvaluation`, logging and path helpers |
| `aeroopt.optimization` | `OptBaseFramework`, `PreProcess` / `PostProcess`, algorithm settings, `Opt*` drivers, MOEA utilities |
| `aeroopt.optimization.stochastic` | Implementations of NSGA-II/III, RVEA, MOEA/D, DE, NRBO, etc. |
| `aeroopt.optimization.hybrid` | `SAO`, `SBO`, and related post-processing |
| `aeroopt.analysis` | Database analysis such as `AnalyzeDatabase` (`aeroopt.analysis.analyze_database`) |
| `aeroopt.utils` | `benchmark`, `surrogate` |

Quick import:

```python
import aeroopt
from aeroopt.core import Problem, Database, MultiProcessEvaluation
from aeroopt.optimization import OptNSGAII, SettingsNSGAII
```

## Examples (`example/`)

Many scripts prepend the repository root to `sys.path` so they run without installing the package; remove that block if you already use `pip install aeroopt`.

| Folder | Script | Summary |
|--------|--------|---------|
| `1-database-io` | `example_core_functions.py` | Problem and database setup; JSON / Excel I/O |
| `2-mp-evaluation` | `example_mpEvaluation.py` | Parallel evaluation with built-in and external executables |
| `3-database-evaluation` | `example_database_evaluation.py` | `Database.evaluate_individuals`: serial vs. multiprocessing, external scripts |
| `4-pre-process` | `example_pre_process.py` | Custom `PreProcess` and candidate-database repair |
| `5-evolutionary-algorithm` | `example_dominance_based_algorithm.py` | `DominanceBasedAlgorithm` dominance and selection tools |
| `5-evolutionary-algorithm` | `example_pareto_analysis.py` | Pareto-focused analysis (pairs with multi-objective examples) |
| `6-single-objective-optimization` | `example_soo.py` | Single-objective comparison: NSGA-II, DE, NRBO, etc. |
| `7-multi-objective-optimization` | `example_nsgaii.py` and others | ZDT benchmarks with NSGA-II, DE, NSGA-III, RVEA, MOEA/D; see `README.md` in that folder |
| `8-surrogate-hybrid-optimization` | `example_sao.py`, `example_sbo.py` | Surrogate-assisted and surrogate-based hybrid optimization |
| `8-surrogate-hybrid-optimization` | `example_kriging.py` | Kriging / surrogate usage |

