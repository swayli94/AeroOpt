Architecture
============

Package layout
--------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Package
     - Role
   * - :mod:`aeroopt.core`
     - ``Problem``, ``Individual``, ``Database``, settings types, parallel evaluation, logging.
   * - :mod:`aeroopt.sampling`
     - Design-of-experiments samplers on the unit hypercube.
   * - :mod:`aeroopt.optimization`
     - The optimization loop, pre/post-processing hooks, genetic operators.
   * - :mod:`aeroopt.optimization.stochastic`
     - NSGA-II, NSGA-III, RVEA, MOEA/D, DE, NRBO.
   * - :mod:`aeroopt.optimization.hybrid`
     - Surrogate-based (SBO) and surrogate-assisted (SAO) frameworks.
   * - :mod:`aeroopt.analysis`
     - Database statistics, crowding metrics, clustering.
   * - :mod:`aeroopt.utils`
     - Benchmark test functions and surrogate-model interfaces.


The three data objects
----------------------

.. code-block:: text

   SettingsData  ─┐
                  ├─►  Problem  ─►  Individual  ─►  Database
   SettingsProblem┘

**Problem** owns everything that is true of the *whole study*: variable names
and bounds, scaling, constraint evaluation, dominance rules, and how to invoke
an external solver. It holds no design data.

**Individual** is one design: its inputs ``x``, its outputs ``y``, whether the
evaluation succeeded, its constraint violations, and the bookkeeping the
algorithms attach to it (Pareto rank, crowding distance, generation, source).
An ``Individual`` keeps a reference to its ``Problem``, so it can scale and
validate itself.

An unevaluated individual has an **empty** ``y`` array rather than ``None``;
:attr:`~aeroopt.core.individual.Individual.is_evaluated` is the check to use.

**Database** is an ordered list of individuals plus the operations a study
needs: duplicate-aware insertion, sorting by several criteria, sub-setting,
merging, shrinking, JSON/Excel round-trips, and dispatching a whole generation
for evaluation.


The four databases
------------------

Every driver maintains four databases with distinct jobs. Understanding which
one an algorithm reads is most of understanding the framework.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Database
     - Contents
   * - ``db_total``
     - **Every** design ever evaluated, including failures and constraint violations. Never shrinks. This is the study's permanent record.
   * - ``db_valid``
     - Rebuilt from ``db_total`` each iteration, keeping only designs that evaluated successfully, lie within bounds, and satisfy all constraints.
   * - ``db_elite``
     - The current answer: the first non-dominated front of ``db_valid``.
   * - ``db_candidate``
     - The generation about to be (or just) evaluated. Transient.

``db_valid`` is deliberately derived rather than maintained incrementally.
Constraints may depend on outputs, and a post-processing hook may change what
counts as feasible --- or prune ``db_total`` outright --- so
:meth:`~aeroopt.optimization.base.OptBaseFramework.derive_valid_from_total`
recomputes it both after the merge and after the hook. An individual the hook
removed is therefore gone from ``db_valid``, from ``db_elite`` and from the
summary of that same iteration.


The iteration loop
------------------

:meth:`~aeroopt.optimization.base.OptBaseFramework.main` runs:

.. code-block:: text

   resume()                              load a previous db_total, if configured
   initialize_population()               sample, pre-process, evaluate, absorb
   select_elite_from_valid()
   save_results()

   while not termination():
       iteration += 1
       update_parameters()               e.g. retrain a surrogate, adapt vectors
       generate_candidate_individuals()  ── algorithm-specific ──
       pre_process.apply()               optional: repair / screen candidates
       evaluate_db_candidate()           snap to grid, drop known designs, evaluate
       update_total_and_valid_with_candidate()
       post_process.apply()              optional: inspect / prune the archive
       derive_valid_from_total()         only when a hook ran
       select_elite_from_valid()
       save_results()

Only ``generate_candidate_individuals`` is mandatory for a new algorithm.
``update_parameters``, ``termination``, ``generate_initial_individuals``,
``save_results`` and ``select_elite_from_valid`` all have working defaults.


Which archive drives the search
-------------------------------

Offspring are normally bred from ``db_valid``. Early in a run, or on a heavily
constrained problem, ``db_valid`` may be nearly empty --- and an algorithm
cannot select or recombine designs it does not have.

:func:`~aeroopt.optimization.base.select_population_database` therefore falls
back to ``db_total`` while

.. math::

   |\text{db\_valid}| \le \max(5,\ 0.5 \cdot N_{\text{pop}})

This is safe because ranking a database that contains infeasible designs uses
the constraint-aware rules of
:meth:`~aeroopt.core.individual.Individual.check_dominance`:

* a feasible individual dominates any infeasible one;
* between two infeasible individuals, neither dominates, and sorting falls back
  to smaller total violation;
* between two feasible individuals, ordinary Pareto dominance applies.

So the early search is driven towards feasibility, and switches to optimizing
objectives once enough feasible designs exist.


Class hierarchy
---------------

.. code-block:: text

   OptBaseFramework                    loop, databases, logging, evaluation
   ├── OptGeneticFramework             + shared SBX/PM settings
   │   ├── OptNSGAII
   │   ├── OptNSGAIII
   │   ├── OptRVEA
   │   └── OptMOEAD
   ├── OptDE
   ├── OptNRBO
   └── SurrogateOptimizationBase       + surrogate training and inner optimizer
       ├── SBO
       └── SAO

Algorithms are kept separate from drivers. An ``Algorithm`` subclass
(``NSGAII``, ``RVEA``, ``MOEAD``, ...) is a stateless collection of static
methods that transforms a population database into a candidate database. The
``Opt*`` driver owns the loop, the databases and the I/O.

That split is what lets SAO run a full ``OptDE`` *inside* one of its own
iterations: the inner optimizer is just another driver, pointed at a surrogate
instead of the real evaluator.


Pre- and post-processing
------------------------

The two hooks are where domain knowledge enters an otherwise generic loop.

:class:`~aeroopt.optimization.base.PreProcess` runs on ``db_candidate`` *before*
evaluation --- the last chance to avoid wasting solver time. It provides helpers
for the common repair patterns:

* ``_restrict_x_values_by_valid_database`` pulls candidates into a distance band
  around known-good designs: too close and the evaluation is redundant, too far
  and a mesh or solver may not converge.
* ``_check_pre_processing_feasibility`` evaluates candidates with a *cheap*
  proxy problem (a geometry check, a coarse model) and flags the failures.
* ``_adjust_x_values_by_valid_database`` repairs only the flagged candidates.

:class:`~aeroopt.optimization.base.PostProcess` runs on ``db_total`` *after*
evaluation, for reporting, archive pruning, or surrogate assessment.
:class:`~aeroopt.optimization.hybrid.sao.PostProcessSAO` uses it to report how
much the surrogate actually contributed each iteration.


Evaluation
----------

:meth:`~aeroopt.core.database.Database.evaluate_individuals` supports three
modes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Behaviour
   * - ``user_func_supports_parallel=True``
     - ``user_func(xs)`` is called once with the whole matrix and returns ``(list_succeed, ys)``, one flag and one row per design, failures included; both are validated before any result is recorded. Use this when the evaluator vectorizes or manages its own parallelism (as a surrogate does).
   * - ``mp_evaluation`` set
     - Each design is submitted to a ``ProcessPoolExecutor``, either as a call to ``user_func`` or as an external run.
   * - neither
     - Serial loop, one design at a time.

In every mode a failed evaluation sets ``valid_evaluation = False``,
``y`` to an empty array and ``sum_violation`` to infinity --- the design is
remembered as failed rather than silently scored as zero.

"Failed" covers every way one design can go wrong, and in all three modes alike:
the evaluator returning ``succeed=False``, the evaluator *raising*, an external
run producing no output file, an external run exceeding its ``timeout``, and a
worker process dying outright. None of them ends the study. Two things still
do, because they are setup mistakes that would fail identically for every
design: :class:`~aeroopt.core.StaleCaseFolderError`, which says the study is
about to read another study's results, and a missing folder name or problem
object for an external run.

A timeout kills the run script *and* the processes it started, and does not
read the case's output file --- see :ref:`solver-timeouts`.


Reproducibility
---------------

Setting ``seed`` in ``SettingsOptimization`` fixes both the initial design of
experiments and the ``numpy.random.Generator`` used by the genetic operators.
A generator can also be injected directly:

.. code-block:: python

   import numpy as np
   opt = OptNSGAII(..., rng=np.random.default_rng(2024))

Every driver accepts ``rng``, and a driver that is not given one derives it from
the settings seed.

.. note::

   Reproducibility covers the search. If the evaluation function is itself
   non-deterministic --- a solver with a wall-clock-dependent convergence path,
   for instance --- the results will still vary.
