# Recipes

## Error to cause

| Message | Cause |
| --------- | ------- |
| `Variable x1**2 ... is not in the data settings` | Constraint string not space-separated. Write `x1 ** 2`. |
| `SettingsX <name> not found in <file>` | The JSON entry's `type` or `name` does not match what the constructor was given. |
| `Name of data settings does not match` | `name_data_settings` differs from the data entry's `name`. |
| `Number of output variables does not match` | `len(output_type)` differs from `len(name_output)`. |
| `MOEA/D: population_size must equal ...` | See the MOEA/D section in settings-reference.md. |
| `OptNRBO only supports single-objective problems` | More than one `output_type` is `±1`. |
| `Cannot build parent database from an empty database` | Every initial design failed or was infeasible. Check the evaluator and constraints. |
| `Surrogate outputs must be a subset of the problem outputs` | The surrogate problem's `name_output` contains a name absent from the main problem. |
| `Invalid ys shape: ...` | The evaluator was handed the whole `xs` matrix: `user_func_supports_parallel=True` but the function takes one design. |
| `Individual problem does not match database problem` | Two different `Problem` objects; they compare equal only when their problem-settings names match. |
| Silent `RuntimeWarning` in numexpr | A constraint divides by a variable that can be zero. |
| `StaleCaseFolderError: Case folder ... a different design` | `Calculation/` holds cases from an earlier study (a new study numbers from 1 again). Clear or move it, set `"resume": true`, or set `problem.rerun_stale_cases = True`. |
| `Calculation folder ... already holds N case folders` | The same thing, warned about before the run spends any solver time. |
| `timeout after ... killing the process tree` | The design exceeded `timeout`; it is recorded as a failure and re-run on the next attempt. |
| `warning: [evaluation] failed for #i` | That design's evaluator raised. It is recorded as a failed design and the study continues. |

## Hung or failing solvers

```python
mp = MultiProcessEvaluation(..., n_process=8,
                            timeout=3600,          # per design
                            batch_timeout=None)    # whole batch; None = no cap
problem.solver_timeout = 3600     # same per-design limit without an mp evaluator
problem.kill_grace_period = 30    # SIGTERM head start before SIGKILL
```

`timeout` bounds **one** evaluation, never the batch — eight processes over
thirty-two designs legitimately take four times one evaluation. On expiry the
run script and everything it started are killed (`SIGTERM`, then `SIGKILL`;
`taskkill /F /T` on Windows), the design is recorded as a failure **without
reading `output.txt`** (a solver that wrote a result and then hung must not pass
it off as valid), and a marker file makes the case re-run rather than be skipped
on a restart.

A job the run script submitted to a batch queue survives the kill, like any
process that puts itself in a new session. Have the script wait for the job so
that killing the script kills the wait.

## Surrogate-based optimization (SBO)

Every candidate comes from an optimization run on the surrogate. Needs
`pip install "aeroopt[surrogate]"`.

```python
from aeroopt.optimization.hybrid import SBO, PostProcessSBO
from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization
from aeroopt.utils.surrogate import Kriging

surrogate = Kriging(problem, train_on_scaled_data=True)

# Inner optimizer: searches the surrogate, never the real evaluator.
inner = OptDE(
    problem=problem,
    optimization_settings=SettingsOptimization('inner_opt', fname_settings=FNAME),
    algorithm_settings=SettingsDE('inner_alg', fname_settings=FNAME),
    logging=False, save_result_files=False,
)

opt = SBO(
    problem=problem,
    optimization_settings=SettingsOptimization('outer_opt', fname_settings=FNAME),
    surrogate=surrogate,
    opt_on_surrogate=inner,
    user_func=evaluate,          # the expensive one
)
opt.post_process = PostProcessSBO(opt, surrogate)   # reports prediction error
opt.main()
```

The inner optimizer needs its own `SettingsOptimization` entry; give it a
larger `population_size` and `max_iterations` than the outer loop, since
surrogate evaluations are free.

## Surrogate-assisted optimization (SAO)

Mixes evolutionary ("E") and surrogate ("S") candidates, so progress continues
while the model is still poor. Same wiring plus `algorithm_settings` (DE for the
evolutionary half) and `ratio_from_surrogate`:

```python
from aeroopt.optimization.hybrid import SAO, PostProcessSAO

opt = SAO(
    problem=problem,
    optimization_settings=SettingsOptimization('outer_opt', fname_settings=FNAME),
    algorithm_settings=SettingsDE('outer_alg', fname_settings=FNAME),
    surrogate=surrogate,
    opt_on_surrogate=inner,
    ratio_from_surrogate=0.5,
    user_func=evaluate,
)
opt.post_process = PostProcessSAO(opt, surrogate)
opt.main()
```

`PostProcessSAO` logs how many "S" vs "E" individuals reached the candidate
front. If "S" never reaches it, lower `ratio_from_surrogate`; if it dominates,
raise it.

## Pre-processing: repair candidates before evaluating

Runs on `db_candidate` before the expensive step — the last chance to avoid
wasting solver time.

```python
from aeroopt.optimization import PreProcess

class RepairCandidates(PreProcess):
    def apply(self) -> None:
        super().apply()                       # logs the candidate count
        xs = self.opt.db_candidate.get_xs()

        # Keep candidates in a useful distance band around known-good designs:
        # too close is redundant, too far may not mesh or converge.
        xs = self._restrict_x_values_by_valid_database(
            xs, min_scaled_distance=0.01, max_scaled_distance=0.20)

        for indi, x in zip(self.opt.db_candidate.individuals, xs):
            indi.update_x(x)      # not `indi.x = x`: refreshes the cached scaled_x

opt.pre_process = RepairCandidates(opt)
```

Other helpers on `PreProcess`:

- `_check_pre_processing_feasibility(xs, cheap_problem, user_func)` — screen
  candidates with a cheap proxy model, returns `(flags, ID_list)`.
- `_adjust_x_values_by_valid_database(xs, flags)` — repair only the flagged ones.

## Post-processing: inspect or prune the archive

Runs on `db_total` after evaluation.

```python
from aeroopt.optimization import PostProcess

class ReportProgress(PostProcess):
    def apply(self) -> None:
        super().apply()
        self.opt.log(f'elite = {self.opt.db_elite.size}', level=1)

opt.post_process = ReportProgress(opt)
```

## Restarting a study

```json
{ "resume": true, "fname_db_resume": "db-resume.json" }
```

Place the previous `db-total.json` at `<working_directory>/Summary/db-resume.json`.
Resumed individuals get `generation = 0` and `source = 'previous_database'`.

A resumed study takes **no** initial sample — it carries on from the designs it
loaded instead of spending another `population_size` evaluations on a fresh
sample of the same space. Set `"force_initial_population_size": 32` to add one
anyway, e.g. to widen a converged archive.

Clear `Calculation/` only if you are *not* resuming: with `"resume": true` the
case folders that match the loaded database are reused, and the rest raise
`StaleCaseFolderError`.

## Analysing an archive offline

```python
from aeroopt.core import Database
from aeroopt.analysis import AnalyzeDatabase

db = Database(problem, database_type='total')
db.read_database_json('Summary/db-total.json')

analyze = AnalyzeDatabase(db)
d_typical, potentials = analyze.calculate_crowding_metrics()
print('mean nearest-neighbour distance:', d_typical)

analyze.calculate_grouping(n_groups=5)      # k-means design families
stats = analyze.calculate_statistics_of_groups()
```

`AnalyzeDatabase` measures crowding in the **input** space (has the design space
been explored evenly), unlike NSGA-II's crowding distance which measures the
objective space.

## Finding lagging trade-off directions

Post-hoc diagnosis of gaps in a computed front:

```python
from aeroopt.optimization.moea import DecompositionBasedAlgorithm as D

ordered, best_g, ref_points = D.find_slow_directions(opt.db_elite, n_partitions=12)
print('worst-covered direction:', ref_points[ordered[0]])
```

Large `best_g` means no design approaches the ideal point along that preference
direction. This is analysis only; no algorithm steers selection towards them.

## Comparing algorithms fairly

Fix the seed and the shared setup, then vary only the driver:

```python
import numpy as np

for name, Driver, Settings in [('nsgaii', OptNSGAII, SettingsNSGAII),
                               ('de',     OptDE,     SettingsDE)]:
    opt = Driver(problem=problem,
                 optimization_settings=SettingsOptimization('opt', fname_settings=FNAME),
                 algorithm_settings=Settings(f'{name}_alg', fname_settings=FNAME),
                 user_func=evaluate,
                 rng=np.random.default_rng(42))
    opt.main()
    print(name, opt.db_elite.size)
```

`example/examples_common.py` does this for the ZDT suite across all MOEAs.

## Benchmarks for testing

```python
from aeroopt.utils import benchmark

benchmark.ZDT1(x)        # 2 objectives, nx >= 2, convex front
benchmark.ZDT2(x)        # non-convex front
benchmark.ZDT3(x)        # disconnected front
benchmark.ZDT4(x)        # many local fronts
benchmark.ZDT6(x)        # non-uniform density
benchmark.Rastrigin(x)   # single objective, many local minima
benchmark.Rosenbrock(x)  # single objective, curved valley
```

All take one `np.ndarray` and return a `float` (single objective) or
`np.ndarray` (multi-objective) — wrap as `lambda x: (True, benchmark.ZDT1(x))`.
