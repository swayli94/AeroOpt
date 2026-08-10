Quickstart
==========

This walkthrough builds a complete two-objective optimization from scratch. The
objective is a plain Python function here; swapping in a CFD solver only changes
that one function (see :ref:`external-evaluation`).

.. note::

   Configuration can be written either in Python or in a JSON file. This page
   uses JSON, which is the better choice for a study worth version-controlling.
   For a self-contained script with no configuration file, see
   :ref:`python-defined-study` at the end, or :doc:`settings`.

1. Describe the data
--------------------

One entry describes the *data* --- the names, bounds and precision of the input
and output variables:

.. code-block:: json

   {
       "zdt_data": {
           "type": "SettingsData",
           "name": "zdt_data",
           "name_input": ["x1", "x2", "x3"],
           "input_low": [0.0, 0.0, 0.0],
           "input_upp": [1.0, 1.0, 1.0],
           "input_precision": [0.0, 0.0, 0.0],
           "name_output": ["y1", "y2"],
           "output_low": [-0.1, -1.0],
           "output_upp": [1.1, 10.0],
           "output_precision": [0.0, 0.0],
           "critical_scaled_distance": 1.0e-8
       }
   }

``input_precision`` snaps a variable to a grid --- useful when a design variable
is a manufacturable quantity such as a 0.1 mm sheet thickness. Use ``0.0`` for
continuous variables.

``critical_scaled_distance`` is the duplicate threshold: two designs closer than
this in the scaled input space are treated as the same design, and the second is
rejected instead of consuming an expensive evaluation.

2. Describe the problem
-----------------------

A second entry turns that data into an optimization *problem* by assigning a
role to each output and listing the constraints:

.. code-block:: json

   {
       "zdt_problem": {
           "type": "SettingsProblem",
           "name": "zdt_problem",
           "name_data_settings": "zdt_data",
           "output_type": [-1, -1],
           "constraint_strings": ["x1 ** 2 + x2 ** 2 - 0.64"]
       }
   }

``output_type`` gives every output one of four roles:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Value
     - Meaning
   * - ``-1``
     - Objective to **minimize**.
   * - ``1``
     - Objective to **maximize**.
   * - ``0``
     - Recorded but not optimized (a monitored quantity).
   * - ``2``
     - Recorded and used for diversity measures, not for dominance.

Constraints are written in the form :math:`g(x, y) \le 0`, so
``"x1 ** 2 + x2 ** 2 - 0.64"`` means :math:`x_1^2 + x_2^2 \le 0.64`. Tokens must
be **separated by spaces**: variable names are substituted by value and the
result is evaluated with ``numexpr``. Constraints that cannot be written as an
expression are supplied as Python callables instead --- see :doc:`settings`.

3. Describe the run
-------------------

.. code-block:: json

   {
       "zdt_opt": {
           "type": "SettingsOptimization",
           "name": "zdt_opt",
           "resume": false,
           "population_size": 32,
           "max_iterations": 20,
           "working_directory": "./",
           "info_level_on_screen": 1,
           "seed": 42
       },
       "zdt_alg": {
           "type": "SettingsNSGAII",
           "name": "zdt_alg",
           "cross_rate": 0.9,
           "mut_rate": 0.9,
           "pow_sbx": 20.0,
           "pow_poly": 20.0
       }
   }

Setting ``seed`` makes the whole run reproducible: it seeds both the initial
design of experiments and the random generator used by the genetic operators.

4. Run it
---------

.. code-block:: python

   import numpy as np

   from aeroopt.core import Problem, SettingsData, SettingsProblem
   from aeroopt.optimization import OptNSGAII, SettingsNSGAII, SettingsOptimization
   from aeroopt.utils import benchmark

   FNAME = 'settings.json'

   def evaluate(x: np.ndarray):
       '''Return (succeed, y). `succeed=False` marks a failed evaluation.'''
       return True, benchmark.ZDT1(x)

   data_settings = SettingsData('zdt_data', fname_settings=FNAME)
   problem_settings = SettingsProblem('zdt_problem', data_settings, fname_settings=FNAME)
   problem = Problem(data_settings, problem_settings)

   opt = OptNSGAII(
       problem=problem,
       optimization_settings=SettingsOptimization('zdt_opt', fname_settings=FNAME),
       algorithm_settings=SettingsNSGAII('zdt_alg', fname_settings=FNAME),
       user_func=evaluate,
   )

   opt.main()

The evaluation function returns ``(succeed, y)``. Returning ``False`` records
the design as a failed evaluation: it stays in ``db_total`` as evidence that the
region is troublesome, but is excluded from ``db_valid`` and from selection.

5. Read the results
-------------------

.. code-block:: python

   # Non-dominated, feasible designs.
   xs = opt.db_elite.get_xs()                       # [n, n_input]
   ys = opt.db_elite.get_ys()                       # [n, n_output]

   # Everything ever evaluated, including failures.
   print('total     :', opt.db_total.size)
   print('feasible  :', opt.db_valid.size)
   print('elite     :', opt.db_elite.size)

   # Persist for later analysis.
   opt.db_total.output_database_json('db-total.json')
   opt.db_total.json_to_excel('db-total.json', 'db-total.xlsx')

With ``save_result_files=True`` (the default) the total and elite databases are
also written to ``<working_directory>/Summary`` after every iteration, so a run
that is killed halfway still leaves usable data behind.

Where the files go
------------------

.. code-block:: text

   working_directory/
   ├── Runfiles/          # template files copied into each external working folder
   ├── Calculation/       # one working folder per external evaluation
   │   ├── 1/
   │   └── 2/
   ├── Summary/           # db-total.json, db-elite.json
   └── optimization.log

.. _external-evaluation:

Evaluating with an external solver
----------------------------------

Leave ``user_func`` as ``None`` and AeroOpt drives an external program instead.
For each design it creates ``Calculation/<ID>/``, copies in the contents of
``Runfiles/``, writes ``input.txt``, and runs ``run.sh`` (``run.bat`` on
Windows) with that folder as the working directory:

.. code-block:: text

   input.txt                 output.txt
   ------------              ------------
                x1  0.35                  y1  0.1225
                x2  0.80                  y2  2.4400

The solver's only contract is to read ``input.txt`` and write ``output.txt`` in
the same ``name value`` format. A missing output file, or a missing variable in
it, is reported as a failed evaluation rather than an exception.

Because a case folder is skipped when its ``input.txt`` already exists, an
interrupted study can be restarted without recomputing finished cases.

Running evaluations in parallel
-------------------------------

.. code-block:: python

   from aeroopt.core import MultiProcessEvaluation

   if __name__ == '__main__':          # required on Windows and macOS spawn
       mp = MultiProcessEvaluation(
           dim_input=problem.n_input,
           dim_output=problem.n_output,
           func=evaluate,              # or None to use the external script
           n_process=8,                # None means serial
       )

       opt = OptNSGAII(..., mp_evaluation=mp)
       opt.main()

The whole generation is submitted to a process pool, so wall-clock time per
iteration approaches the cost of the slowest single evaluation.

.. _python-defined-study:

The same study without a settings file
--------------------------------------

Every settings class can be built directly from Python, which keeps a quick
experiment in a single script:

.. code-block:: python

   import numpy as np

   from aeroopt.core import Problem, SettingsData, SettingsProblem
   from aeroopt.optimization import OptNSGAII, SettingsNSGAII, SettingsOptimization
   from aeroopt.utils import benchmark

   def evaluate(x: np.ndarray):
       return True, benchmark.ZDT1(x)

   data_settings = SettingsData.from_values(
       'zdt_data',
       name_input=['x1', 'x2', 'x3'],
       input_low=[0.0, 0.0, 0.0], input_upp=[1.0, 1.0, 1.0],
       name_output=['y1', 'y2'],
       output_low=[-0.1, -1.0], output_upp=[1.1, 10.0],
   )

   problem_settings = SettingsProblem.from_values(
       'zdt_problem', data_settings,
       output_type=[-1, -1],
       constraint_strings=['x1 ** 2 + x2 ** 2 - 0.64'],
   )

   opt = OptNSGAII(
       problem=Problem(data_settings, problem_settings),
       optimization_settings=SettingsOptimization.from_values(
           'zdt_opt', population_size=32, max_iterations=20, seed=42),
       algorithm_settings=SettingsNSGAII.from_values(
           'zdt_alg', cross_rate=0.9, mut_rate=0.9),
       user_func=evaluate,
   )
   opt.main()

Omitted keys take their documented defaults, so only what differs has to be
stated. When the configuration is worth keeping, write it out and switch to the
file-based form above:

.. code-block:: python

   from aeroopt.core import save_settings

   save_settings([data_settings, problem_settings, opt.optimization_settings],
                 'settings.json')

Next steps
----------

* :doc:`settings` --- every configuration key, and custom constraint callables.
* :doc:`architecture` --- how the databases and the iteration loop fit together.
* :doc:`principles/index` --- what each algorithm actually computes.
* :doc:`examples` --- the runnable scripts in ``example/``.
