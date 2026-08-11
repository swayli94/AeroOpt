---
name: aeroopt
description: >-
  Sets up and runs optimization studies with the aeroopt framework: writing the
  settings.json, defining a Problem, choosing among NSGA-II / NSGA-III / RVEA /
  MOEA-D / DE / NRBO / SBO / SAO, wiring a Python or external-executable
  evaluator, adding a Kriging surrogate, and reading the resulting databases.
  Use when working in the AeroOpt repository, when the user mentions aeroopt,
  SettingsData, SettingsProblem, OptNSGAII or another Opt* driver, db_elite or
  db_valid, or asks to set up a multi-objective / surrogate-assisted
  optimization, connect a CFD or FEA solver to an optimizer, or debug an
  aeroopt run.
---

# AeroOpt

`aeroopt` optimizes designs whose evaluation is expensive (a CFD run, an FEA
solve, an external executable). It keeps every evaluation in a persistent
`Database` and exposes hooks to intervene before an expensive evaluation is
spent.

## Non-obvious facts

These cause most failures. Read them before writing any aeroopt code.

1. **Settings come from a JSON file *or* from Python — both are first class.**
   `SettingsX('name', fname_settings='settings.json')` matches a JSON entry by
   `type` (the class name) **and** `name`. `SettingsX.from_values('name', ...)`
   builds the same object from keyword arguments with no file at all. Use
   Python for quick experiments and notebooks, JSON for studies worth
   version-controlling; `save_settings` converts the first into the second.

2. **Constraint string tokens must be separated by spaces.**
   `"x1 ** 2 + x2 ** 2 - 0.64"` works. `"x1**2 + x2**2 - 0.64"` raises
   `ValueError: Variable x1**2 ... is not in the data settings`, because the
   parser splits on spaces and substitutes variable names.

3. **Constraints are `g(x, y) <= 0`.** To express `y1 >= 40`, write
   `"40.0 - y1"`. Both inputs and outputs may appear.

4. **`output_low` / `output_upp` are a scale, not a limit.** Their only consumer
   is `scale_y`, an affine map that does **not** clip — an out-of-range `y` is
   scaled outside `[0, 1]` and nothing raises. Dominance, crowding and NSGA-III
   all renormalize and barely care, but **MOEA/D and RVEA do**: their weight
   vectors and reference-vector angles assume the objectives span comparable
   fractions of `[0, 1]`. Set fixed bounds covering the plausible range with a
   small margin; never widen them "to be safe".

5. **Never clip an evaluated `y`.** `apply_bounds_y` clips in place and the
   framework never calls it — clipping collapses distinct designs onto one
   objective value (a fake tie in dominance) and stores a number the solver
   never returned. Use it only on values the code *constructed*: a surrogate
   extrapolation, a sampled output, a fixed-axis plot. An unacceptable result is
   `succeed=False` or a constraint, never a clip.

6. **`output_precision` quantizes `y` before scaling**, so it sets the noise
   floor every comparison sees, not just what is stored. Leave it `0.0` unless
   the solver's own noise justifies a grid — and keep it well below the bound
   span, or the output is deactivated and scales to a constant `0.0`.

7. **`user_func` returns `(succeed, y)`**, not `y`. Return `False` to record a
   failed evaluation; it stays in `db_total` but is excluded from `db_valid`.

8. **MOEA/D requires `population_size` == the number of Das-Dennis reference
   points**, which is `comb(p + M - 1, M - 1)` for `M` objectives and
   `n_partitions = p`; for 2 objectives that is `p + 1`. A mismatch raises at
   construction.

9. **NRBO is single-objective only.** It raises otherwise.

10. **`MultiProcessEvaluation` needs `if __name__ == '__main__':`** in the entry
    script, as any Python multiprocessing does.

## Minimal working study, defined in Python

No settings file involved. Prefer this when the user just wants to run
something.

```python
import numpy as np
from aeroopt.core import Problem, SettingsData, SettingsProblem
from aeroopt.optimization import OptNSGAII, SettingsOptimization, SettingsNSGAII

def evaluate(x: np.ndarray):
    """Return (succeed, y)."""
    return True, np.array([x[0], 1.0 - np.sqrt(x[0]) + x[1]])

data_settings = SettingsData.from_values(
    'demo_data',
    name_input=['x1', 'x2'], input_low=[0.0, 0.0], input_upp=[1.0, 1.0],
    name_output=['f1', 'f2'], output_low=[-0.1, -1.0], output_upp=[1.1, 10.0],
)

problem_settings = SettingsProblem.from_values(
    'demo_problem', data_settings,
    output_type=[-1, -1],                       # both minimized
    constraint_strings=['x1 ** 2 + x2 ** 2 - 0.64'],
)

problem = Problem(data_settings, problem_settings)

opt = OptNSGAII(
    problem=problem,
    optimization_settings=SettingsOptimization.from_values(
        'demo_opt', population_size=32, max_iterations=20, seed=42),
    algorithm_settings=SettingsNSGAII.from_values(
        'demo_alg', cross_rate=0.9, mut_rate=0.9),
    user_func=evaluate,
)
opt.main()

xs = opt.db_elite.get_xs()   # non-dominated feasible designs
ys = opt.db_elite.get_ys()
```

Omitted keys take their documented defaults, so `from_values` only has to state
what differs. `input_precision` / `output_precision` default to continuous, and
`name_data_settings` defaults to the `data_settings` passed in.

## The same study, defined in JSON

Prefer this when the configuration should be version-controlled or shared.

```python
data_settings = SettingsData('demo_data', fname_settings='settings.json')
problem_settings = SettingsProblem('demo_problem', data_settings,
                                   fname_settings='settings.json')
```

To go from Python to a file:

```python
from aeroopt.core import save_settings

save_settings([data_settings, problem_settings, opt_settings], 'settings.json')
```

The file needs four entries. Generate a correct, runnable pair with the
scaffolding script rather than writing it by hand:

```bash
python skills/aeroopt/scripts/new_study.py --help
python skills/aeroopt/scripts/new_study.py \
    --dir my_study --n-input 3 --objectives 2 --algorithm nsgaii
```

It writes `my_study/settings.json` + `my_study/run.py` and validates that the
settings load. For the full key reference see
[settings-reference.md](settings-reference.md).

## The four databases

| Database | Contents |
| ---------- | ---------- |
| `db_total` | Every design ever evaluated, **including failures**. Never shrinks. |
| `db_valid` | Rebuilt each iteration: successful, in-bounds, constraint-satisfying only. |
| `db_elite` | The answer: first non-dominated front of `db_valid`. |
| `db_candidate` | The generation being evaluated. Transient. |

Reading results: `get_xs()`, `get_ys()`, `get_unified_objectives()`,
`output_database_json(path)`, `json_to_excel(json_path, xlsx_path)`.

A row of `get_ys()` is zero for a failed individual; check
`Individual.is_evaluated` rather than assuming every row is real.

## Choosing an algorithm

| Situation | Use |
| ----------- | ----- |
| 2-3 objectives, general use | `OptNSGAII` |
| 4+ objectives | `OptNSGAIII` or `OptRVEA` |
| Objectives on very different scales | `OptRVEA` (adapts its reference vectors) |
| Continuous variables, rugged landscape | `OptDE` |
| Single objective | `OptNRBO` or `OptDE` |
| Explicit scalarized subproblems | `OptMOEAD` |
| Evaluation very expensive, model trusted | `SBO` |
| Evaluation very expensive, model unproven | `SAO` |

`OptRVEA`'s diversity schedule depends on `max_iterations`; set it to the budget
actually intended, and do not stop the run early expecting the same behaviour.

## External solver

Leave `user_func=None`. For each design aeroopt creates
`Calculation/<ID>/`, copies in the **contents** of `Runfiles/`, writes
`input.txt`, and runs `run.sh` (`run.bat` on Windows) with that folder as the
working directory.

```text
Runfiles/            input.txt              output.txt
├── run.sh           ---------              ----------
└── solver_files       x1  0.35               y1  0.1225
                       x2  0.80               y2  2.4400
```

The solver's whole contract is: read `input.txt`, write `output.txt` in the same
`name value` format. A missing file or variable is reported as a failed
evaluation, not an exception.

Case IDs are unique for the whole run, so a design keeps one `<ID>` across
`Calculation/`, `db-total.json` and the log.

A case folder whose `input.txt` already holds the **same** design is skipped, so
an interrupted study restarts without recomputing finished cases. One holding a
*different* design is a leftover from an earlier study (a new study numbers from
1 again) and raises `StaleCaseFolderError` rather than returning the old
`output.txt` for the new design — clear or move `Calculation/`, set
`"resume": true`, or set `problem.rerun_stale_cases = True` to re-run those
folders.

## Parallel evaluation

```python
from aeroopt.core import MultiProcessEvaluation

if __name__ == '__main__':
    mp = MultiProcessEvaluation(
        dim_input=problem.n_input,
        dim_output=problem.n_output,
        func=evaluate,      # or None for the external script
        n_process=8,        # None means serial
    )
    opt = OptNSGAII(..., mp_evaluation=mp)
    opt.main()
```

Set `user_func_supports_parallel=True` instead when the evaluator itself takes
the whole `xs` matrix and returns `(list_succeed, ys)` — `ys` with one row per
design and `list_succeed` with one flag per design, **failures included**. Both
are checked before any result is recorded.

## Hung and failing solvers

`timeout=3600` bounds **one** external evaluation, not the batch (eight
processes over thirty-two designs legitimately take four times one evaluation).
Without a parallel evaluator, set `problem.solver_timeout = 3600` instead.
On expiry the run script and every process it started are killed — `SIGTERM`,
then `SIGKILL` after `problem.kill_grace_period` — and the design is recorded as
a **failure without reading `output.txt`**, so a solver that wrote a result and
then hung cannot pass that value off as valid. A marker file makes the case
re-run instead of being skipped on a restart. A job the script submitted to a
batch queue escapes the kill; have the script wait for it so that killing the
script kills the wait.

`batch_timeout` is the opt-in hard cap on the whole batch; when it fires the
designs still running are recorded as failures and the study continues.

An evaluation that raises, or a worker that dies, is recorded as a failed design
rather than ending the study. The exceptions are setup mistakes:
`StaleCaseFolderError` and a missing folder name / problem object still stop it.

## Reproducibility

Set `seed` in the optimization settings, or pass `rng=np.random.default_rng(n)`
to any driver. Both cover the initial sample **and** the genetic operators.

## More

- [settings-reference.md](settings-reference.md) — every JSON key, the output
  bounds / precision / clipping strategy in full, constraint callables, and how
  to add a settings class.
- [recipes.md](recipes.md) — surrogate optimization, pre/post-processing hooks,
  restarting a study, analysing an archive, and the error-to-cause table.

## House rules for this repository

- Run `pytest` after changing `aeroopt/`; the suite is fast (~3 s).
- Docstrings are `'''...'''` with NumPy-style `Parameters:` / `Returns:`
  sections. Sphinx builds with `cd docs && make strict` (warnings are errors).
- Keep `docs/source/changelog.rst` current for user-visible changes.
- New algorithms subclass `Algorithm` (stateless static operators) plus an
  `Opt*` driver subclassing `OptBaseFramework` or `OptGeneticFramework`. Only
  `generate_candidate_individuals` is mandatory.
