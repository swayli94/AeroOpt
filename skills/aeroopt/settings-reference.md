# Settings reference

Settings can be built two equivalent ways. Both run the same conversion,
defaulting and validation.

## From Python

```python
data_settings = SettingsData.from_values(
    'demo_data',
    name_input=['x1', 'x2'], input_low=[0.0, 0.0], input_upp=[1.0, 1.0],
    name_output=['f1'], output_low=[0.0], output_upp=[1.0],
)

# Equivalently, with a mapping:
data_settings = SettingsData('demo_data', settings={'name_input': [...], ...})
```

Only the keys that differ from the defaults need stating. Classes taking extra
constructor arguments keep them positional:

```python
problem_settings = SettingsProblem.from_values(
    'demo_problem', data_settings, output_type=[-1])
```

## From a JSON file

One file holds every entry. The top-level key is an arbitrary label; what
identifies an entry is `type` (the settings class name) and `name`:

```python
SettingsData('demo_data', fname_settings='settings.json')
#             ^^^^^^^^^ matched against the entry's "name" field
```

A template with every entry ships as `aeroopt/template_settings.json`.

## Python to file

```python
from aeroopt.core import save_settings

save_settings([data_settings, problem_settings, opt_settings], 'settings.json')
```

Entries are keyed `<type>_<name>` unless `entry_names=[...]` is given. The
result reads back through the constructors unchanged, so a study prototyped
inline can be frozen into a file for the record.

## Complete example

```json
{
    "data": {
        "type": "SettingsData",
        "name": "demo_data",
        "name_input": ["x1", "x2", "x3"],
        "input_low": [0.0, 0.0, 0.0],
        "input_upp": [1.0, 1.0, 1.0],
        "input_precision": [0.0, 0.0, 0.0],
        "name_output": ["y1", "y2"],
        "output_low": [-0.1, -1.0],
        "output_upp": [1.1, 10.0],
        "output_precision": [0.0, 0.0],
        "critical_scaled_distance": 1.0e-8
    },
    "problem": {
        "type": "SettingsProblem",
        "name": "demo_problem",
        "name_data_settings": "demo_data",
        "output_type": [-1, -1],
        "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"]
    },
    "opt": {
        "type": "SettingsOptimization",
        "name": "demo_opt",
        "resume": false,
        "population_size": 32,
        "max_iterations": 20,
        "working_directory": "./",
        "info_level_on_screen": 1,
        "seed": 42
    },
    "alg": {
        "type": "SettingsNSGAII",
        "name": "demo_alg",
        "cross_rate": 0.9,
        "mut_rate": 0.9,
        "pow_sbx": 20.0,
        "pow_poly": 20.0
    }
}
```

`name_data_settings` in the problem entry **must** equal the `name` of the data
entry, otherwise construction raises.

## SettingsData

| Key | Type | Meaning |
| ----- | ------ | --------- |
| `name_input` | list[str] | Input names; length defines `n_input`. |
| `input_low` / `input_upp` | list[float] | Bounds; swapped automatically if inverted. |
| `input_precision` | list[float] | Grid the variable snaps to; `0.0` = continuous. |
| `name_output` | list[str] | Output names; length defines `n_output`. |
| `output_low` / `output_upp` | list[float] | **Scaling** bounds, not constraints. |
| `output_precision` | list[float] | Output grid; `0.0` = continuous. |
| `critical_scaled_distance` | float | Duplicate threshold in scaled input space. |

A variable whose range is smaller than its precision is **deactivated**: held at
its lower bound and ignored in distances. This is the supported way to freeze a
variable without restructuring the configuration.

### Output bounds, precision and clipping

`output_low` / `output_upp` are **not constraints**. Their only consumer is
`Problem.scale_y`, an affine map that does **not** clip: a `y` outside the
bounds is scaled outside `[0, 1]` and nothing raises. The choice only decides
how comparable the objectives are to each other, and only part of the machinery
is sensitive to it:

| Consumer | Sensitivity | Why |
| ---------- | ------------- | ----- |
| MOEA/D decomposition + ideal point | decisive | Weight vectors spread evenly over the simplex, assuming comparable spans. |
| RVEA angle-penalized distance | decisive | Niching is by *angle* in scaled space — pure geometry. |
| Kriging with `train_on_scaled_data` | moderate | Training targets and cross-output `epistemic_std` are scaled by them. |
| Output named in `sample_variables` | moderate | There the bounds really are the sampled interval. |
| Pareto dominance / non-dominated sorting | none | Monotone affine per objective; ordering unchanged. |
| Crowding distance | none | Renormalized by each front's own span. |
| NSGA-III niching | none in practice | Renormalized by ideal point + hyperplane intercepts. |
| Single-objective DE / NRBO | none | Ranking one monotone-transformed value. |

So NSGA-II/III tolerate a rough guess; MOEA/D and RVEA do not.

**Choosing them.** Prefer fixed, hand-chosen bounds. MOEA/D's ideal point and
subproblem slots and RVEA's reference vectors persist across generations, so a
scale drifting with the data would compare this generation against the last on a
different ruler; fixed bounds also keep a resumed study, and two studies of the
same problem, comparable.

- Physics first: a `Cd` known to lie in `[0.005, 0.05]` needs no measurement.
- Otherwise run a small DOE, read `statistics['min_y']` / `['max_y']` from
  `AnalyzeDatabase`, and cover the observed range with a 10–20 % margin. A
  placeholder like `±1e6` left in place is not a choice.
- Balance beats width: if one objective sweeps 0.8 of `[0, 1]` and another
  0.02, MOEA/D's weights buy nothing — tighten the second, don't grow the
  population.

**`output_precision`** is applied to `y` *before* scaling, so it quantizes the
values every comparison sees, not just what is stored. Setting it to the
solver's noise level deliberately stops the search chasing unresolvable
differences; `0.0` (continuous) is the right default otherwise. Note that
`output_low` / `output_upp` are themselves snapped to the grid at construction,
and an output whose bound span is below its precision is deactivated to a
constant `0.0`.

**Clipping.** `check_bounds_y` reports; `apply_bounds_y` clips in place. The
framework calls neither, deliberately: clipping collapses distinct designs onto
one objective value — a fabricated tie in dominance, and a stored `y` the solver
never returned, undetectable downstream. An out-of-bounds `y` is comparatively
harmless (it costs MOEA/D and RVEA some resolution, nothing more).

- Value out of range means the design is unacceptable → `succeed=False` from the
  evaluator, or a constraint `g(x, y) <= 0`. Not a clip.
- Clip only what the code *constructed* rather than measured: a surrogate
  extrapolating to a physically impossible value before an infill criterion that
  assumes `[0, 1]`, an output sampled or interpolated as a design condition, a
  fixed-axis plot. Ask whether this `y` is an observation or a number you
  invented; never clip the first.

## SettingsProblem

| Key | Type | Meaning |
| ----- | ------ | --------- |
| `name_data_settings` | str | Must equal the data entry's `name`. |
| `output_type` | list[int] | Role of each output, see below. |
| `constraint_strings` | list[str] | Expressions read as `g(x, y) <= 0`. |

`output_type` values:

| Value | Meaning |
| ------- | --------- |
| `-1` | Objective to minimize |
| `1` | Objective to maximize |
| `0` | Recorded only, not optimized |
| `2` | Recorded, used for diversity measures, not dominance |

### Constraint strings

Tokens **must be space-separated**; variable names are substituted by value and
the expression is evaluated with `numexpr`.

```text
"x1 ** 2 + x2 ** 2 - 0.64"   ->  x1² + x2² <= 0.64      correct
"x1**2 + x2**2 - 0.64"       ->  raises ValueError      wrong
"40.0 - y1"                  ->  y1 >= 40               correct
"y1 - x1"                    ->  y1 <= x1               correct
```

### Constraint callables

Anything not expressible as a string becomes a Python callable returning the
violation value:

```python
from aeroopt.core import CustomConstraintFunction

class MinimumThickness(CustomConstraintFunction):
    """Require y1 >= 40, i.e. 40 - y1 <= 0."""
    def __call__(self, x, y) -> float:
        return float(40.0 - y[0])

problem_settings.constraint_functions.append(MinimumThickness(data_settings))
```

Every constraint contributes `max(0, g)` to `sum_violation`. An individual with
`sum_violation > 0` is infeasible: kept in `db_total`, excluded from `db_valid`.

## SettingsOptimization

| Key | Default | Meaning |
| ----- | --------- | --------- |
| `resume` | `false` | Load `fname_db_resume` instead of sampling fresh. |
| `population_size` | `64` | Individuals per generation. |
| `max_iterations` | `100` | Generations after the initial population. |
| `working_directory` | `"./"` | Root of `Calculation`, `Summary`, `Runfiles`. |
| `info_level_on_screen` | `1` | Messages at or below this level print; all are logged. |
| `critical_potential_x` | `0.2` | Potential at the typical neighbour distance. |
| `seed` | `null` | Seeds the initial sample **and** the operators. |
| `force_initial_population_size` | `null` | Overrides `population_size` for the first generation only. A resumed study samples nothing unless this is set; `0` skips the sample in a fresh study. |
| `fname_db_total` | `"db-total.json"` | Written to `Summary`. |
| `fname_db_elite` | `"db-elite.json"` | Written to `Summary`. |
| `fname_db_resume` | `"db-resume.json"` | Read when `resume` is true. |
| `fname_log` | `"optimization.log"` | Relative to `working_directory`. |

Unrecognized keys are set verbatim as attributes, so a subclass can read its own
options from the same entry.

## Algorithm settings

NSGA-II, NSGA-III, RVEA and MOEA/D share four keys:

| Key | Default | Meaning |
| ----- | --------- | --------- |
| `cross_rate` | `1.0` | Probability of crossing a parent pair. |
| `mut_rate` | `1.0` | Expected number of mutated variables per individual; divided by `n_input` internally. |
| `pow_sbx` | `20.0` | SBX distribution index; larger keeps children near parents. |
| `pow_poly` | `20.0` | Polynomial mutation index; larger gives smaller steps. |

Per-algorithm additions:

| Class | Key | Default | Meaning |
| ------- | ----- | --------- | --------- |
| `SettingsNSGAIII` | `n_partitions` | `null` | Das-Dennis grid; inferred from `population_size` when null. |
| `SettingsRVEA` | `n_partitions` | `null` | As above. |
| | `alpha` | `2.0` | APD penalty exponent on progress; larger favours convergence. |
| | `adapt_freq` | `0.1` | Fraction of `max_iterations` between vector adaptations. |
| `SettingsMOEAD` | `n_partitions` | `null` | Das-Dennis grid. |
| | `n_neighbors` | `20` | Neighbourhood size `T`. |
| | `prob_neighbor_mating` | `0.9` | Probability of mating within the neighbourhood. |
| | `decomposition` | `"auto"` | `"tchebicheff"`, `"pbi"`, or `"auto"`. |
| | `pbi_theta` | `5.0` | PBI penalty on perpendicular distance. |
| `SettingsDE` | `scale_factor` | `0.5` | Differential weight `F`. |
| | `cross_rate` | `0.8` | Binomial crossover rate `CR`. |
| `SettingsNRBO` | `deciding_factor` | `0.6` | Probability of the trap avoidance operator. |

### MOEA/D population size

`population_size` must equal `comb(p + M - 1, M - 1)` for `M` objectives and
`n_partitions = p`. For two objectives that is `p + 1`:

```python
from aeroopt.optimization.moea import DecompositionBasedAlgorithm as D

p = D.suggest_n_partitions(n_objective=2, population_size=32)
n = D.das_dennis_reference_points(2, p).shape[0]   # set population_size to n
```

## Adding a settings class

All settings classes derive from `SettingsBase`, which handles both
construction paths, entry matching, conversion, defaults and export. Declare
fields once:

```python
from aeroopt.optimization.settings import (
    REQUIRED, FieldSpec, SettingsGeneticOperators,
)

class SettingsMyAlgorithm(SettingsGeneticOperators):
    """Settings of my algorithm."""

    _FIELDS: tuple[FieldSpec, ...] = SettingsGeneticOperators._FIELDS + (
        ('my_parameter', float, 0.5),
        ('required_one', str, REQUIRED),
    )
```

Each entry is `(attribute_name, converter, default)`; `REQUIRED` makes a key
mandatory. The JSON `type` is matched against the class name automatically, and
the class immediately supports `from_values`, `settings=`, `to_dict` and
`save_settings` with no extra code.

Set `_ALLOW_EXTRA_KEYS = True` to have undeclared keys set verbatim as
attributes (as `SettingsOptimization` does).
