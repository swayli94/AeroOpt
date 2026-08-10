Dominance and Pareto fronts
===========================

The problem with "better"
-------------------------

With one objective, comparing two designs is trivial. With several conflicting
objectives, most pairs are simply not comparable: a wing with lower drag but
also lower lift is neither better nor worse than its rival.

Pareto dominance is the standard way to make that precise. For minimization,
design :math:`a` **dominates** :math:`b` when it is at least as good everywhere
and strictly better somewhere:

.. math::

   a \prec b \iff
   \forall i:\ f_i(a) \le f_i(b)
   \quad\wedge\quad
   \exists j:\ f_j(a) < f_j(b)

:meth:`~aeroopt.core.problem.Problem.check_pareto_dominance` returns four
outcomes:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - Meaning
   * - ``1``
     - ``y1`` dominates ``y2``.
   * - ``-1``
     - ``y1`` is dominated by ``y2``.
   * - ``9``
     - Non-dominated: each is better in some objective. **The interesting case.**
   * - ``0``
     - Equal on every objective.

Non-dominated designs are exactly the trade-offs a designer has to choose
between, which is why a multi-objective run returns a *set* rather than a single
answer. That set is the **Pareto front**.


Non-dominated sorting
---------------------

:meth:`~aeroopt.optimization.moea.DominanceBasedAlgorithm.non_dominated_ranking`
partitions a population into layers:

* **Front 1** --- designs dominated by nobody. The current best trade-off surface.
* **Front 2** --- designs dominated only by front 1.
* **Front k** --- and so on.

The algorithm is Deb's: for each individual *p*, count how many dominate it
(:math:`n_p`) and record which ones it dominates (:math:`S_p`). Everything with
:math:`n_p = 0` forms front 1. Removing that front decrements the counters of
everything it dominated; whatever reaches zero forms front 2, and so on.

Complexity is :math:`O(MN^2)` for *N* individuals and *M* objectives. On the
expensive-evaluation problems this framework targets, that cost is irrelevant
next to a single CFD run.

Each individual's ``pareto_rank`` is set in place (1-based), and the fronts are
cached on the database as ``index_pareto_fronts``.

.. note::

   Ranking a database whose ``is_valid_database`` flag is set uses the fast
   vectorized comparison of scaled objectives. Otherwise it falls back to
   :meth:`~aeroopt.core.individual.Individual.check_dominance`, which is
   constraint-aware. This is what makes it safe to breed from ``db_total``
   before enough feasible designs exist.


Why rank alone is not enough
----------------------------

Rank tells you which designs are good, not which are *diverse*. A population
that has collapsed onto one corner of the Pareto front can be entirely rank-1
while telling the designer almost nothing.

Worse, selection pressure actively causes this: crowded regions produce more
offspring, which crowd the region further. Every algorithm in this package
therefore pairs a convergence measure (rank, or distance to the ideal point)
with a diversity measure. The choice of diversity measure is what distinguishes
them:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Algorithm
     - Diversity measure
   * - NSGA-II
     - Crowding distance --- objective-space spacing to immediate neighbours.
   * - NSGA-III
     - Niche count on predefined reference directions.
   * - RVEA
     - Angle to the nearest reference vector, penalized over time.
   * - MOEA/D
     - Built in: one solution slot per weight vector.
   * - Adaptive sampling
     - Potential field over the *input* space, see :doc:`crowding`.


Elites
------

:meth:`~aeroopt.optimization.moea.DominanceBasedAlgorithm.select_elite_from_valid`
copies front 1 of ``db_valid`` into ``db_elite`` after every iteration, sorted
by ID. That database is the deliverable: the set of feasible designs for which
no evaluated alternative is better in every respect.
