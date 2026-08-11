Changelog
=========

0.2.1
-----

Correctness fixes for external evaluation: working folders are no longer reused
between iterations, a stale folder is reported instead of silently read back,
and the input precision grid now holds for every design that reaches the solver.

Fixed
^^^^^

* **External working folders were reused every iteration.** A candidate is
  evaluated in ``Calculation/<ID>``, but ``db_candidate`` is emptied and
  refilled each iteration, so its IDs restarted at 1 every time. From the first
  iteration on, every candidate found the initial population's ``input.txt``
  already in place, skipped the solver, and read back the *previous* design's
  ``output.txt``. The wrong ``y`` was then stored against the new ``x`` --- and
  since the two designs really are different, the duplicate check let it into
  ``db_total``. Candidates are now numbered uniquely for the whole run
  (:meth:`~aeroopt.optimization.base.OptBaseFramework._assign_ID_to_candidate_individuals`
  was written for this but never called), from a counter that only moves
  forward, so a candidate rejected as a duplicate cannot hand its number to a
  later design. A case now also keeps the same ID in ``Calculation/``, in
  ``db-total.json`` and in the log.
* **A stale case folder is now an error, not a silent result.**
  ``Problem.external_run`` skipped the solver whenever ``input.txt`` existed. It
  now compares the recorded design with the requested one and raises
  :class:`~aeroopt.core.StaleCaseFolderError` when they differ --- the case
  belongs to an earlier study, whether because a new study numbers from 1 again
  or because a re-parameterization changed what the variable names mean. Set
  ``problem.rerun_stale_cases = True`` to re-prepare and re-run such folders
  instead. Matching folders are still skipped, so restarting an interrupted
  study is unchanged. A driver about to run into a populated ``Calculation``
  folder says so before spending any solver time.
* **The pre-processing feasibility check reused its folders too.** Its cases are
  numbered from 1 on every call, so ``Calculation/PreProcess/<n>`` collided
  across iterations; the names are now prefixed with the iteration.
* **Offspring ignored** ``input_precision``. It was applied only by
  ``scale_x``, i.e. to the initial sample. Every trial vector from SBX,
  polynomial mutation, DE and NRBO was continuous, so a variable declared as
  integer-like reached the solver as a fraction --- with binomial crossover, on
  roughly ``cross_rate`` of all candidates, which is most of a DE generation.
  The operators now call the new :meth:`~aeroopt.core.Problem.apply_precision_x`
  after applying the bounds, and the driver snaps ``db_candidate`` once more
  before evaluating, which also covers candidates a pre-processing hook
  produced.
* ``Problem.latin_hypercube_sampling(sample_variables=[...])`` scaled the named
  variables by hand and skipped the precision grid that the whole-vector path
  applies through ``scale_x``.
* **MOEA/D rebound its subproblem slots to the wrong designs.** The neighbour
  replacement queued ``(subproblem, offspring_ID)`` during generation and
  resolved the ID after evaluation, but the ID an offspring carries at
  generation time is not the one it ends up with. The queue now holds the
  individual itself.

Changed
^^^^^^^

* ``MOEAD.generate_candidate_individuals`` fills ``pending_list`` with
  ``(subproblem_index, Individual)`` instead of ``(subproblem_index, ID)``.
* :class:`~aeroopt.core.Problem` gained ``rerun_stale_cases`` (default False)
  and :meth:`~aeroopt.core.Problem.apply_precision_x`;
  :class:`~aeroopt.core.Individual` gained
  :meth:`~aeroopt.core.Individual.update_x`, which replaces ``x`` and refreshes
  the cached ``scaled_x`` that the duplicate check reads.

0.2.0
-----

Audit of the whole package: correctness fixes, deduplication, and this
documentation.

Fixed
^^^^^

* **Packaging.** ``aeroopt/utils/`` had no ``__init__.py``, so setuptools
  excluded it from the built distribution. An installed ``aeroopt`` was missing
  ``aeroopt.utils.benchmark`` and ``aeroopt.utils.surrogate``, which made
  ``aeroopt.optimization.hybrid`` fail to import.
* **Undeclared dependencies.** ``numexpr``, ``pydoe`` and ``openpyxl`` are
  imported at runtime but were not in ``dependencies``. ``smt`` is now an
  optional ``surrogate`` extra.
* **Python 3.9 compatibility.** Twelve modules used ``X | None`` annotations
  without ``from __future__ import annotations``, so importing the package
  raised ``TypeError`` on the minimum supported Python.
* **External runs.** ``Problem.external_run`` copied the run-files *folder*
  into each case directory instead of its contents, so the run script was never
  where it was expected and every external evaluation failed. The shell-string
  calls were replaced by ``shutil.copytree`` and ``subprocess.run``, which also
  makes ``timeout`` work on all platforms and handles paths containing spaces.
  Its platform check was ``platform.system() in 'Windows'``, a substring test.
* **Failed evaluations crashed the optimizer.** A failed evaluation leaves an
  individual with an empty output vector in ``db_total``, and the drivers breed
  from ``db_total`` whenever the valid archive is still small. Reading
  objectives from such a database raised a broadcast error, and
  ``Individual.check_dominance`` indexed the empty vector --- so a single
  diverging solver run took the whole study down, in exactly the situation this
  framework exists for. ``Database.get_ys`` now leaves those rows at zero, and
  dominance treats an unevaluated individual as worse than any evaluated one
  (two failures being mutually non-dominated).
* **In-place mutation.** ``Problem.scale_x`` / ``scale_y`` applied precision
  rounding to the caller's array, silently modifying inputs passed for scaling.
* ``Problem`` defined ``__eq__`` without ``__hash__`` and was therefore
  unhashable.
* ``Problem.read_input`` / ``read_output`` raised ``IndexError`` on blank or
  malformed lines instead of reporting a failed evaluation.
* **Crowding analysis.** ``AnalyzeDatabase.eliminate_crowding_individuals``
  always raised, because the crowding-distance flag it sorts on was never set.
* ``AnalyzeDatabase.calculate_potential_induced_by_database`` tested a float
  attribute against ``None``, so the potential coefficient stayed at 0 and every
  distance scored 1.0.
* ``AnalyzeDatabase`` raised a broadcast error on any database containing a
  failed evaluation, because those individuals carry an empty output array.
* ``Database.copy_from_database(deepcopy=False)`` assigned the source's list
  object, so the two databases shared one list and adding to either changed
  both.
* ``Database.get_sub_database()`` with no selector returned an empty database
  although it documented "all individuals".
* ``Database.delete_individual`` left the Pareto rank and cached fronts marked
  as current.
* ``Database.shrink_database`` used ``<`` instead of ``<=`` and could index past
  the start of the list when the tail was fully reserved.
* **Mutable default arguments.** ``SettingsProblem.constraint_functions``
  defaulted to a shared ``[]``, so constraints appended to one problem leaked
  into every later problem. ``MOEAD.generate_candidate_individuals`` defaulted
  to a shared list and a ``Generator`` built at import time.
* ``SAO`` computed a surrogate prediction vector and discarded it, and could
  delete surrogate candidates it had just added once the eviction index went
  negative.
* ``PostProcessSAO`` computed performance metrics over empty index arrays when
  one candidate source was unused.
* ``Kriging`` did not call ``super().__init__``.
* :class:`~aeroopt.optimization.hybrid.sbo.SBO` defaulted
  ``user_func_supports_parallel=True`` while the other seven drivers defaulted
  to ``False``, so an ordinary per-design evaluator was handed the whole ``xs``
  matrix and failed with ``Invalid ys shape``. The flag concerns the expensive
  evaluator only; the inner optimizer's own flag is set internally.

Added
^^^^^

* **Settings can be defined in Python, not only in JSON.** Every settings class
  now accepts ``SettingsX.from_values('name', key=value, ...)`` or
  ``SettingsX('name', settings=mapping)``, running the same conversion,
  defaulting and validation as the file path. A complete study can be written in
  one script with no ``settings.json`` on disk.
* :func:`~aeroopt.core.settings_base.save_settings` and
  :meth:`~aeroopt.core.settings_base.SettingsBase.to_dict` export
  Python-defined settings to a file the constructors read back unchanged, so a
  prototype can be frozen into a reproducible artifact.
* :class:`~aeroopt.core.settings_base.SettingsBase` moved to
  :mod:`aeroopt.core.settings_base` and is now the base of *every* settings
  class, including :class:`~aeroopt.core.settings.SettingsData` and
  :class:`~aeroopt.core.settings.SettingsProblem`, which previously hand-wrote
  their loading. The sentinel is exported publicly as
  :data:`~aeroopt.core.settings_base.REQUIRED`.
* ``input_precision`` / ``output_precision`` may now be omitted, defaulting to
  continuous variables, and ``name_data_settings`` defaults to the name of the
  ``data_settings`` passed to :class:`~aeroopt.core.settings.SettingsProblem`.

Changed
^^^^^^^

* ``aeroopt.core.mpEvaluation`` renamed to ``aeroopt.core.mp_evaluation``, and
  ``template_usr_func`` to ``template_user_func``.

  .. warning::

     This is a breaking change: the old module path was removed rather than
     kept as an alias. Update imports to

     .. code-block:: python

        from aeroopt.core.mp_evaluation import MultiProcessEvaluation

     Importing ``MultiProcessEvaluation`` from :mod:`aeroopt.core` was, and
     remains, the recommended form and is unaffected.
* ``Database.critical_scaled_distance`` is now a property, matching
  ``Problem.critical_scaled_distance``.
* ``Individual.y`` is always an ndarray; use the new ``is_evaluated`` property
  instead of comparing against ``None``. Sort types are named constants
  (``SORT_BY_ID``, ``SORT_BY_CROWDING``, ...).
* All settings classes derive from ``SettingsBase`` and declare their fields
  declaratively, replacing seven near-identical ``read_settings`` methods.
* ``OptRVEA`` and ``OptMOEAD`` accept ``rng``, like every other driver. All
  drivers derive an unseeded generator from ``SettingsOptimization.seed``, so a
  seed in the settings file now makes the entire run reproducible.
* The duplicated "use ``db_valid`` unless it is too small" rule became
  ``select_population_database``; ``select_elite_from_valid`` became a base-class
  default; the shared SBX/mutation offspring loop became
  ``fill_candidates_by_sbx_and_mutation``; the SAO/SBO shared plumbing became
  ``SurrogateOptimizationBase``.
* ``Algorithm``'s optional hooks return working defaults instead of ``None``.
* ``aeroopt.sampling``, previously an empty package, now holds the
  design-of-experiments samplers that ``Problem`` delegates to.
* ``aeroopt.analysis`` and ``aeroopt.utils`` export their public names.
* ``benchmark.Gussian`` warns; use ``Gaussian``.

Removed
^^^^^^^

* ``reserve_ratio`` from the NSGA-II / NSGA-III / RVEA settings and
  ``fname_db_population`` from the optimization settings: both were parsed but
  never used. Existing settings files keep working; the keys are ignored.
