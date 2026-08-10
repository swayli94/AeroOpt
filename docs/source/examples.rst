Examples
========

The ``example/`` directory holds runnable scripts, ordered so that each builds
on the previous one. They generate their own settings files and clean up after
themselves, so they can be run directly:

.. code-block:: bash

   cd example/7-multi-objective-optimization
   python example_nsgaii.py

Each script prepends the repository root to ``sys.path`` so it works from a
clone without installing. Delete that block if you installed the package.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder / script
     - What it shows
   * - ``1-database-io/example_core_functions.py``
     - Building a ``Problem`` and ``Database`` from scratch, adding individuals, custom constraint callables, JSON and Excel round-trips.
   * - ``2-mp-evaluation/example_mpEvaluation.py``
     - ``MultiProcessEvaluation`` with a Python function and with an external script, on Linux and Windows.
   * - ``3-database-evaluation/example_database_evaluation.py``
     - ``Database.evaluate_individuals`` in all three modes, verifying serial and parallel results agree.
   * - ``4-pre-process/example_pre_process.py``
     - A custom ``PreProcess`` that repairs candidates against the valid archive.
   * - ``5-evolutionary-algorithm/example_dominance_based_algorithm.py``
     - Non-dominated sorting, crowding distance, parent selection and database shrinking, with plots.
   * - ``5-evolutionary-algorithm/example_pareto_analysis.py``
     - Post-hoc analysis of lagging reference directions on a computed front.
   * - ``6-single-objective-optimization/example_soo.py``
     - Comparison of NSGA-II, DE and NRBO on a single-objective problem.
   * - ``7-multi-objective-optimization/example_*.py``
     - NSGA-II, NSGA-III, RVEA, MOEA/D and DE on the ZDT suite with a shared setup, so the figures are directly comparable.
   * - ``8-surrogate-hybrid-optimization/example_kriging.py``
     - Fitting a Kriging model and inspecting its predictions and uncertainty.
   * - ``8-surrogate-hybrid-optimization/example_sbo.py``
     - Surrogate-based optimization with an inner DE loop.
   * - ``8-surrogate-hybrid-optimization/example_sao.py``
     - Surrogate-assisted optimization mixing evolutionary and surrogate candidates.


Comparing algorithms fairly
---------------------------

``example/examples_common.py`` fixes the problem size, population size,
iteration count, plot limits and random seeds shared by the multi-objective
examples. Because every run starts from the same initial population and uses
aligned random streams, differences between the figures reflect the algorithms
rather than luck.

.. code-block:: python

   from examples_common import (
       MAX_ITERATIONS, N_INPUT, POPULATION_SIZE,
       PLOT_F1_LIM, PLOT_F2_LIM_BY_BENCHMARK,
       apply_benchmark_seeds,
   )

   apply_benchmark_seeds(bench_index)   # identical initial population per benchmark


A minimal external solver
-------------------------

The evaluation examples generate a small external "solver" to stand in for a
real one. It is the complete contract: read ``input.txt``, write ``output.txt``.

.. code-block:: python

   # Runfiles/external_evaluator.py
   from pathlib import Path

   def main():
       cwd = Path.cwd()
       values = {}
       for line in (cwd / 'input.txt').read_text(encoding='utf-8').splitlines():
           parts = line.split()
           if len(parts) >= 2:
               values[parts[0]] = float(parts[1])

       y1 = values['x1'] ** 2 + values['x2'] ** 2

       with (cwd / 'output.txt').open('w', encoding='utf-8') as f:
           f.write(f'y1 {y1}\n')

   if __name__ == '__main__':
       main()

.. code-block:: bash

   # Runfiles/run.sh
   python external_evaluator.py

Everything in ``Runfiles/`` is copied into each case folder before the run, and
``run.sh`` (or ``run.bat``) is executed with that folder as the working
directory. Replace the Python script with a call to your solver and nothing else
changes.
