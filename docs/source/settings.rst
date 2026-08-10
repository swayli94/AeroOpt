Settings reference
==================

Settings objects can be built two equivalent ways: from a JSON file, or
directly in Python. Both run the same conversion, defaulting and validation, so
pick whichever suits the task and convert between them freely.

Defining settings in Python
---------------------------

:meth:`~aeroopt.core.settings_base.SettingsBase.from_values` takes the same keys
a JSON entry would hold, as keyword arguments:

.. code-block:: python

   from aeroopt.core import Problem, SettingsData, SettingsProblem

   data_settings = SettingsData.from_values(
       'zdt_data',
       name_input=['x1', 'x2', 'x3'],
       input_low=[0.0, 0.0, 0.0],
       input_upp=[1.0, 1.0, 1.0],
       name_output=['y1', 'y2'],
       output_low=[-0.1, -1.0],
       output_upp=[1.1, 10.0],
   )

   problem_settings = SettingsProblem.from_values(
       'zdt_problem', data_settings,
       output_type=[-1, -1],
       constraint_strings=['x1 ** 2 + x2 ** 2 - 0.64'],
   )

   problem = Problem(data_settings, problem_settings)

Only keys that differ from the defaults need stating. ``input_precision`` and
``output_precision`` default to continuous, and ``name_data_settings`` defaults
to the name of the ``data_settings`` passed in.

A mapping works too, which is convenient when the values are computed:

.. code-block:: python

   fields = {'name_input': names, 'input_low': lower, 'input_upp': upper, ...}
   data_settings = SettingsData('zdt_data', settings=fields)

This path involves no file at all --- useful in notebooks, in tests, and when
sweeping a configuration programmatically.

Defining settings in a JSON file
--------------------------------

All configuration lives in a single file. Each top-level key is an arbitrary
label; what identifies an entry is its ``type`` (the settings class name) and
its ``name`` (the value passed to the constructor):

.. code-block:: python

   SettingsData('zdt_data', fname_settings='settings.json')
   #             ^^^^^^^^^ matched against the entry's "name"

.. code-block:: json

   {
       "any_label_you_like": {
           "type": "SettingsData",
           "name": "zdt_data"
       }
   }

Keeping data, problem, optimization and algorithm settings in one file means a
run is described by one artifact that can be version-controlled next to its
results.

A template with every supported entry ships with the package as
``aeroopt/template_settings.json``.

Converting Python settings to a file
------------------------------------

:func:`~aeroopt.core.settings_base.save_settings` writes settings objects back
out as a file the constructors can read:

.. code-block:: python

   from aeroopt.core import save_settings

   save_settings([data_settings, problem_settings, opt_settings],
                 'settings.json')

Entries are keyed ``<type>_<name>`` unless ``entry_names`` is given. The result
round-trips exactly, so a study prototyped inline can be frozen into a file once
it is worth keeping.

.. tip::

   Prototype in Python; commit JSON. The Python path keeps a quick experiment
   in one file, and ``save_settings`` turns it into the reproducible artifact
   once the configuration stabilizes.


SettingsData
------------

Describes the design variables and outputs.

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Type
     - Meaning
   * - ``name_input``
     - list[str]
     - Names of the input variables. Length defines ``n_input``.
   * - ``input_low`` / ``input_upp``
     - list[float]
     - Lower and upper bounds. Swapped automatically if inverted.
   * - ``input_precision``
     - list[float]
     - Grid each variable is snapped to; ``0.0`` means continuous.
   * - ``name_output``
     - list[str]
     - Names of the output variables. Length defines ``n_output``.
   * - ``output_low`` / ``output_upp``
     - list[float]
     - Bounds used to scale outputs to ``[0, 1]``.
   * - ``output_precision``
     - list[float]
     - Output grid; ``0.0`` means continuous.
   * - ``critical_scaled_distance``
     - float
     - Duplicate threshold in scaled input space.

.. admonition:: Why output bounds matter
   :class: note

   Output bounds are not constraints. They define the scaling used to compare
   objectives with each other. Dominance, crowding distance, reference-point
   niching and decomposition all operate on scaled objectives, so a drag
   coefficient in ``[0.01, 0.05]`` and a lift coefficient in ``[0.1, 1.5]``
   contribute comparably. Bounds that are far too wide compress all the real
   variation into a sliver of ``[0, 1]`` and weaken diversity preservation.

A variable whose range is smaller than its precision is treated as
**deactivated**: it is held at its lower bound and contributes nothing to
distances. This is the supported way to freeze a variable without editing the
rest of the configuration.


SettingsProblem
---------------

Turns data into an optimization problem.

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Type
     - Meaning
   * - ``name_data_settings``
     - str
     - Must equal the ``name`` of the matching ``SettingsData`` entry.
   * - ``output_type``
     - list[int]
     - Role of each output: ``-1`` minimize, ``1`` maximize, ``0`` recorded only, ``2`` diversity.
   * - ``constraint_strings``
     - list[str]
     - Expressions read as :math:`g(x, y) \le 0`.

Constraint strings
^^^^^^^^^^^^^^^^^^

Tokens **must be separated by spaces**. Variable names are replaced by their
numeric values (wrapped in parentheses, so negative values keep the expected
precedence) and the result is evaluated with ``numexpr``:

.. code-block:: text

   "x1 ** 2 + x2 ** 2 - 0.64"     ->  x1² + x2² <= 0.64      correct
   "x1**2 + x2**2 - 0.64"         ->  parsed as one token    wrong
   "y1 - x1"                      ->  y1 <= x1               correct

Both inputs and outputs may appear, so a constraint can depend on the result of
the evaluation.

Constraint functions
^^^^^^^^^^^^^^^^^^^^

Anything that is not a simple expression is supplied as a Python callable
returning the violation value:

.. code-block:: python

   from aeroopt.core import CustomConstraintFunction

   class MinimumThickness(CustomConstraintFunction):
       '''Require y1 >= 40, i.e. 40 - y1 <= 0.'''
       def __call__(self, x, y) -> float:
           return float(40.0 - y[0])

   problem_settings.constraint_functions.append(MinimumThickness(data_settings))

Every constraint contributes ``max(0, g)`` to the individual's
``sum_violation``. An individual with ``sum_violation > 0`` is infeasible: it
is kept in ``db_total`` but excluded from ``db_valid``.


SettingsOptimization
--------------------

Controls the optimization loop, independently of the algorithm.

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 44

   * - Key
     - Type
     - Default
     - Meaning
   * - ``resume``
     - bool
     - ``false``
     - Load ``fname_db_resume`` instead of starting from a new sample.
   * - ``population_size``
     - int
     - ``64``
     - Individuals per generation.
   * - ``max_iterations``
     - int
     - ``100``
     - Generations after the initial population.
   * - ``working_directory``
     - str
     - ``"./"``
     - Root of ``Calculation``, ``Summary`` and ``Runfiles``.
   * - ``info_level_on_screen``
     - int
     - ``1``
     - Messages at or below this level are echoed to the screen; everything is logged.
   * - ``critical_potential_x``
     - float
     - ``0.2``
     - Potential at the typical neighbour distance, see :doc:`principles/crowding`.
   * - ``seed``
     - int or null
     - ``null``
     - Seeds the initial sample and the operators; set it for reproducible runs.
   * - ``force_initial_population_size``
     - int or null
     - ``null``
     - Overrides ``population_size`` for the first generation only. Set to ``0`` to skip initial sampling entirely, e.g. when resuming.
   * - ``fname_db_total``
     - str
     - ``"db-total.json"``
     - Total database, written to ``Summary``.
   * - ``fname_db_elite``
     - str
     - ``"db-elite.json"``
     - Elite database, written to ``Summary``.
   * - ``fname_db_resume``
     - str
     - ``"db-resume.json"``
     - Database read when ``resume`` is true.
   * - ``fname_log``
     - str
     - ``"optimization.log"``
     - Log file, relative to ``working_directory``.

Unrecognized keys are set verbatim as attributes, so a subclass can read its own
options from the same entry.


Algorithm settings
------------------

NSGA-II, NSGA-III, RVEA and MOEA/D share the SBX / polynomial-mutation operator
pair, and therefore share four keys:

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Key
     - Default
     - Meaning
   * - ``cross_rate``
     - ``1.0``
     - Probability of applying crossover to a parent pair.
   * - ``mut_rate``
     - ``1.0``
     - Expected number of mutated variables per individual. Divided by ``n_input`` internally, so ``1.0`` means "about one variable".
   * - ``pow_sbx``
     - ``20.0``
     - SBX distribution index. Larger keeps children closer to their parents.
   * - ``pow_poly``
     - ``20.0``
     - Polynomial mutation distribution index. Larger gives smaller steps.

Per-algorithm keys:

.. list-table::
   :header-rows: 1
   :widths: 22 22 12 44

   * - Class
     - Key
     - Default
     - Meaning
   * - ``SettingsNSGAIII``
     - ``n_partitions``
     - ``null``
     - Das-Dennis grid resolution; inferred from ``population_size`` when null.
   * - ``SettingsRVEA``
     - ``n_partitions``
     - ``null``
     - As above.
   * -
     - ``alpha``
     - ``2.0``
     - APD penalty exponent on search progress. Larger favours convergence.
   * -
     - ``adapt_freq``
     - ``0.1``
     - Fraction of ``max_iterations`` between reference-vector adaptations.
   * - ``SettingsMOEAD``
     - ``n_partitions``
     - ``null``
     - Das-Dennis grid resolution.
   * -
     - ``n_neighbors``
     - ``20``
     - Neighbourhood size *T* of each subproblem.
   * -
     - ``prob_neighbor_mating``
     - ``0.9``
     - Probability of mating within the neighbourhood.
   * -
     - ``decomposition``
     - ``"auto"``
     - ``"tchebicheff"``, ``"pbi"``, or ``"auto"``.
   * -
     - ``pbi_theta``
     - ``5.0``
     - PBI penalty on the perpendicular distance.
   * - ``SettingsDE``
     - ``scale_factor``
     - ``0.5``
     - Differential weight *F*.
   * -
     - ``cross_rate``
     - ``0.8``
     - Binomial crossover rate *CR*.
   * - ``SettingsNRBO``
     - ``deciding_factor``
     - ``0.6``
     - Probability of applying the trap avoidance operator.

.. warning::

   MOEA/D requires ``population_size`` to equal the number of Das-Dennis
   reference points for the chosen ``n_partitions``. That count is
   :math:`\binom{p + M - 1}{M - 1}` for *M* objectives and *p* partitions, so
   for two objectives it is simply :math:`p + 1`. A mismatch raises at
   construction time rather than failing silently mid-run.


Adding your own settings class
------------------------------

All settings classes derive from
:class:`~aeroopt.core.settings_base.SettingsBase`, which handles both
construction paths, entry matching, type conversion, defaults and export. A new
algorithm declares its fields once:

.. code-block:: python

   from aeroopt.optimization.settings import (
       REQUIRED, FieldSpec, SettingsGeneticOperators,
   )

   class SettingsMyAlgorithm(SettingsGeneticOperators):
       '''Settings of my algorithm.'''

       _FIELDS: tuple[FieldSpec, ...] = SettingsGeneticOperators._FIELDS + (
           ('my_parameter', float, 0.5),
           ('required_one', str, REQUIRED),
       )

Each entry is ``(attribute_name, converter, default)``;
:data:`~aeroopt.core.settings_base.REQUIRED` marks a key as mandatory. The JSON
``type`` is matched against the class name automatically, and the class
immediately supports ``from_values``, ``settings=``, ``to_dict`` and
``save_settings`` with no further code.

Set ``_ALLOW_EXTRA_KEYS = True`` to have undeclared keys set verbatim as
attributes, as :class:`~aeroopt.optimization.settings.SettingsOptimization`
does.
