# AGENTS.md

Development principles for AeroOpt. Applies to humans and AI agents alike.

For *using* the library, see `skills/aeroopt/SKILL.md`. This file is about
*changing* it.

## What this project is for

AeroOpt optimizes designs whose evaluation is expensive — a CFD run, an FEA
solve, an external executable that takes minutes to hours per sample. Two
consequences drive every design decision:

1. **An evaluation is precious.** Wasting one on a duplicate, an out-of-bounds
   design, or a candidate a cheap check would have rejected is a real cost.
2. **Evaluations fail.** Solvers diverge, meshes break, licences time out. A
   failure must be *recorded*, never crash the study and never be silently
   scored as zero.

When a change trades generality for either of these, take the trade.

## Ground rules

- **Run `pytest` before claiming anything works.** The suite is ~3 s. There is
  no excuse for skipping it.
- **A bug fix comes with a regression test** that fails before the fix. Put it
  in `tests/test_regressions.py` with a docstring saying what broke and why it
  mattered.
- **Keep `docs/` and `skills/` truthful.** They are checked into the repo and
  are the first thing a user (or an agent) reads. A change that makes them wrong
  is not finished.
- **Report honestly.** If tests fail, say so and show the output. If a step was
  skipped, say which. Never describe unverified work as done.

## Architecture

```
aeroopt/
├── core/            Problem, Individual, Database, settings, evaluation
├── sampling/        design-of-experiments samplers on the unit hypercube
├── optimization/    the loop, operators, pre/post-processing
│   ├── stochastic/  NSGA-II/III, RVEA, MOEA/D, DE, NRBO
│   └── hybrid/      SBO, SAO
├── analysis/        crowding metrics, statistics, clustering
└── utils/           benchmarks, surrogate interfaces
```

Dependencies point **downwards only**: `optimization` may import `core`, never
the reverse. `core` must not import `optimization` or `analysis`.

### Algorithms are separate from drivers

An `Algorithm` subclass (`NSGAII`, `RVEA`, `MOEAD`, …) is a **stateless**
collection of static methods that turns a population database into a candidate
database. An `Opt*` driver owns the loop, the four databases and the I/O.

That separation is what lets `SAO` run a whole `OptDE` *inside* one of its own
iterations. Do not put run state on an `Algorithm`.

To add an algorithm you implement `generate_candidate_individuals` and subclass
`OptBaseFramework` (or `OptGeneticFramework` if it uses SBX + polynomial
mutation). Everything else — sampling, evaluation, archive updates, elite
selection, logging, restart — has a working default.

### The four databases

`db_total` keeps everything ever evaluated including failures and never shrinks.
`db_valid` is **derived** from it each iteration, not maintained incrementally,
because constraints can depend on outputs and a post-processing hook can change
what counts as feasible. `db_elite` is the first non-dominated front of
`db_valid`. `db_candidate` is transient.

If you find yourself updating `db_valid` in place, you are working against the
design.

## Coding conventions

- Docstrings are `'''...'''` with NumPy-style `Parameters:` / `Returns:`
  sections. Sphinx renders them; `cd docs && make strict` treats warnings as
  errors.
- `from __future__ import annotations` at the top of every module. The package
  supports Python 3.9, where `X | None` in a runtime-evaluated annotation is a
  `TypeError` without it.
- Type-annotate public signatures. Prefer `List[str]` from `typing` for
  consistency with the existing code.
- Names say what the thing is: `neighbor_slots`, not `Nloc`;
  `objectives_neighbors`, not `F_nei`. Single letters are acceptable only for
  loop indices and where they match a cited paper's notation (`d1`, `d2`,
  `theta` in the PBI formula).
- `ruff check aeroopt/ tests/ --select F,E4,E7,E9,W,B --ignore E501` must be
  clean.

### Comments explain *why*

The code already says what it does. A comment earns its place by explaining a
non-obvious reason:

```python
# `dirs_exist_ok` copies the *contents* of the run-files folder. A plain
# `cp -r Runfiles folder/` would nest it as a subdirectory, because the case
# folder was just created above.
shutil.copytree(self.runfiles_folder, folder, dirs_exist_ok=True)
```

Do not restate the line below it.

## Traps this codebase has already fallen into

Each of these was a real bug. Do not reintroduce them.

**Mutable default arguments.** `constraint_functions=[]` leaked constraints
between problems; `pending_list=[]` and `rng=np.random.default_rng()` were built
once at import. Default to `None` and construct inside.

**In-place mutation of a caller's array.** `SettingsData.apply_precision` works
in place, so `scale_x` used to silently round the array the caller passed in.
Copy before transforming.

**Assuming every individual has outputs.** A failed evaluation leaves `y` as an
**empty array**, and `db_total` normally holds several. Guard with
`Individual.is_evaluated`; never index `y` unconditionally.

**Forgetting `__init__.py`.** `aeroopt/utils/` had none, so setuptools dropped
it from the wheel and the published package could not import. Every package
directory needs one, and `tests/test_regressions.py` now asserts it.

**Undeclared imports.** `numexpr`, `pydoe` and `openpyxl` were imported but
absent from `dependencies`. If you import it, declare it.

**Stale build artifacts.** `build/` caches copied sources, so a deleted module
can survive into a fresh wheel. `rm -rf build aeroopt.egg-info` before packaging.

**Inconsistent defaults across siblings.** `SBO` defaulted
`user_func_supports_parallel=True` while the other seven drivers defaulted to
`False`, which made an ordinary evaluator fail. When adding a parameter that
already exists elsewhere, match it.

## Settings

Every settings class derives from `SettingsBase` and declares its fields once:

```python
class SettingsMyAlgorithm(SettingsGeneticOperators):
    _FIELDS = SettingsGeneticOperators._FIELDS + (
        ('my_parameter', float, 0.5),
    )
```

`(attribute, converter, default)`; use `REQUIRED` as the default to make a key
mandatory. Loading, conversion, defaulting and validation are inherited — do
**not** hand-write a `read_settings`.

Both construction paths must keep working for every class:

```python
SettingsMyAlgorithm('name', fname_settings='settings.json')   # from JSON
SettingsMyAlgorithm.from_values('name', my_parameter=0.7)     # from Python
```

`save_settings([...], 'settings.json')` writes Python-defined settings back out,
so the two directions round-trip. `tests/test_settings_from_python.py` enforces
this.

## Backward compatibility

The package is `0.x` and alpha, so breaking changes are allowed — but they are
never silent:

- Record them in `docs/source/changelog.rst` with the migration.
- Keep the old spelling working with a `DeprecationWarning` when the fix is
  cheap.
- Removing a public name outright is a deliberate decision, not a side effect of
  a refactor.

Settings **files** are the most user-visible surface. Adding a key with a
default is safe. Removing or renaming one breaks every existing study, so
prefer ignoring an obsolete key over rejecting it.

## Reproducibility

`seed` in `SettingsOptimization` covers the initial sample **and** the genetic
operators. Every driver accepts `rng`. A new algorithm that draws randomness
must thread the driver's `self.rng` through — never call
`np.random.default_rng()` inside an operator without an escape hatch, and never
use the legacy global `np.random.*` in library code.

## Checklist for a change

- [ ] `pytest` passes.
- [ ] Bug fixes have a regression test that fails without the fix.
- [ ] `ruff check aeroopt/ tests/ --select F,E4,E7,E9,W,B --ignore E501` clean.
- [ ] `cd docs && make strict` passes (warnings are errors).
- [ ] New/changed public API is documented and appears in the API reference.
- [ ] User-visible changes are in `docs/source/changelog.rst`.
- [ ] `skills/aeroopt/` still describes reality.
- [ ] Examples still run, if you touched anything they exercise.
