AeroOpt
=======

**AeroOpt** (``aeroopt``) is a multi-objective and single-objective optimization
framework built for engineering workflows where evaluating a design is the
expensive part: a CFD run, an FEA solve, or any external executable that takes
minutes to hours per sample.

That premise drives every design choice in the package:

* **Every evaluation is kept.** Designs accumulate in a persistent
  :class:`~aeroopt.core.database.Database` that can be written to JSON or Excel,
  reloaded, merged, sub-setted, and restarted from. The optimizer works on an
  archive, not on a transient generation.
* **Evaluation is pluggable and parallel.** An objective can be a Python
  callable or an external solver invoked in its own working folder;
  :class:`~aeroopt.core.mp_evaluation.MultiProcessEvaluation` spreads a
  generation across processes.
* **The loop is open for intervention.** Every iteration exposes
  :class:`~aeroopt.optimization.base.PreProcess` and
  :class:`~aeroopt.optimization.base.PostProcess` hooks, so candidates can be
  repaired, screened by a cheap surrogate check, or replaced by hand before the
  expensive evaluation is spent on them.
* **Failure is expected.** A solver that diverges or a design that violates a
  constraint is recorded rather than dropped, and the search adapts to how much
  feasible data actually exists.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   settings

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   architecture
   principles/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   examples
   api/index
   changelog


At a glance
-----------

.. code-block:: python

   import numpy as np
   from aeroopt.core import Problem, SettingsData, SettingsProblem
   from aeroopt.optimization import OptNSGAII, SettingsOptimization, SettingsNSGAII

   def evaluate(x: np.ndarray):
       """Return (succeed, y). Anything can happen in here: a solver, a script, a model."""
       return True, np.array([x[0], 1.0 - np.sqrt(x[0]) + x[1]])

   data_settings = SettingsData('demo', fname_settings='settings.json')
   problem_settings = SettingsProblem('demo', data_settings, fname_settings='settings.json')
   problem = Problem(data_settings, problem_settings)

   opt = OptNSGAII(
       problem=problem,
       optimization_settings=SettingsOptimization('demo', fname_settings='settings.json'),
       algorithm_settings=SettingsNSGAII('demo', fname_settings='settings.json'),
       user_func=evaluate,
   )
   opt.main()

   print(opt.db_elite.size, 'non-dominated designs')


Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Algorithm
     - Driver
     - Selection principle
   * - NSGA-II
     - :class:`~aeroopt.optimization.stochastic.nsgaii.OptNSGAII`
     - Non-dominated rank, then crowding distance within a front
   * - NSGA-III
     - :class:`~aeroopt.optimization.stochastic.nsgaiii.OptNSGAIII`
     - Non-dominated rank, then niching onto Das-Dennis reference points
   * - RVEA
     - :class:`~aeroopt.optimization.stochastic.rvea.OptRVEA`
     - Angle-penalized distance to adaptive reference vectors
   * - MOEA/D
     - :class:`~aeroopt.optimization.stochastic.moead.OptMOEAD`
     - Scalarized subproblems with neighbourhood replacement
   * - Differential evolution
     - :class:`~aeroopt.optimization.stochastic.de.OptDE`
     - DE/rand/1/bin offspring on a rank-and-crowding archive
   * - NRBO
     - :class:`~aeroopt.optimization.stochastic.nrbo.OptNRBO`
     - Newton-Raphson search rule with trap avoidance (single objective)
   * - SBO
     - :class:`~aeroopt.optimization.hybrid.sbo.SBO`
     - All candidates from an optimization run on a surrogate
   * - SAO
     - :class:`~aeroopt.optimization.hybrid.sao.SAO`
     - Evolutionary and surrogate-derived candidates mixed per iteration

For richer visualization and multi-criteria decision making around a computed
Pareto set, see `pymoo <https://pymoo.org/>`_.


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
