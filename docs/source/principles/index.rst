Principles
==========

These pages explain what the framework actually computes: how designs are
compared, how offspring are produced, and how each algorithm decides who
survives.

.. toctree::
   :maxdepth: 2

   dominance
   operators
   dominance_algorithms
   decomposition_algorithms
   surrogate
   crowding


The common ground
-----------------

Every multi-objective algorithm here answers the same two questions each
generation:

1. **Who reproduces?** A parent pool is truncated out of the archive.
2. **What do they produce?** Variation operators turn parents into candidates.

They differ almost entirely in the *first* answer --- the selection principle.
NSGA-II ranks by dominance and breaks ties with crowding distance; NSGA-III
breaks ties by niching onto reference points; RVEA scores by angle-penalized
distance; MOEA/D replaces the notion of a population front with a set of
scalarized subproblems. The variation step is shared: three of the four use the
same SBX plus polynomial mutation pair.


Scaled objectives
-----------------

All comparisons happen on **scaled, direction-unified** objectives.

*Scaled* means each output is mapped to :math:`[0, 1]` using the bounds in
``SettingsData``:

.. math::

   \tilde{y}_i = \frac{y_i - y_i^{\text{low}}}{y_i^{\text{upp}} - y_i^{\text{low}}}

Without this, an objective measured in Pascals would dominate every distance
computation against one measured in radians.

The map is affine and does **not** clip, so an evaluation outside the bounds is
simply scaled outside :math:`[0, 1]`. How much the choice of bounds matters
depends on the algorithm: dominance, crowding distance and NSGA-III niching all
renormalize and are insensitive to it, whereas MOEA/D's weight vectors and
RVEA's reference-vector angles assume the objectives span comparable fractions
of :math:`[0, 1]`. See :ref:`output-bounds-strategy` for how to pick them, why
the framework never clips an evaluated result, and what ``output_precision``
does to the values every comparison sees.

*Direction-unified* means maximization objectives are negated, so that in
:meth:`~aeroopt.core.database.Database.get_unified_objectives` **smaller is
always better**. Every algorithm downstream can therefore assume minimization.

.. math::

   f_i = \begin{cases}
       \tilde{y}_i  & \text{if output } i \text{ is minimized (type } -1)\\
       -\tilde{y}_i & \text{if output } i \text{ is maximized (type } +1)
   \end{cases}


Constraints
-----------

For constraints written as :math:`g_k(x, y) \le 0`, the total violation is

.. math::

   V = \sum_k \max(0,\ g_k(x, y))

so satisfied constraints contribute nothing and cannot offset a violated one.
A design is feasible when :math:`V = 0`.

Constraint handling is embedded in the dominance relation itself
(:meth:`~aeroopt.core.individual.Individual.check_dominance`) rather than in a
penalty added to the objectives:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Situation
     - Result
   * - feasible vs. infeasible
     - the feasible one dominates
   * - infeasible vs. infeasible
     - non-dominated; sorting prefers smaller :math:`V`
   * - feasible vs. feasible
     - ordinary Pareto dominance on objectives

The advantage over a penalty is that it needs no penalty weight, and the
objective values of an infeasible design --- which may be meaningless --- never
influence the comparison.
