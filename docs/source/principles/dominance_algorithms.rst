Dominance-based algorithms
==========================

NSGA-II, NSGA-III and RVEA all rank by Pareto dominance first. They differ in
how they break ties inside the front that does not fit --- and that difference
is what makes one work where another fails.


NSGA-II
-------

:class:`~aeroopt.optimization.stochastic.nsgaii.NSGAII`

Fronts are filled in order until the next one would overflow the population.
Inside that last partial front, individuals are ranked by **crowding distance**:
the perimeter of the box spanned by a design's immediate neighbours in each
objective.

.. math::

   d_i = \sum_{m=1}^{M} \frac{f_m^{(i+1)} - f_m^{(i-1)}}{f_m^{\max} - f_m^{\min}}

Boundary points get :math:`d = \infty`, which permanently protects the extremes
of the front from being crowded out. Larger distance means a lonelier design, so
selection prefers it.

Crowding distance is cheap, parameter-free and needs no prior knowledge of the
front's shape --- which is why NSGA-II remains the default choice for two and
three objectives.

**Where it breaks down.** In :math:`M` objectives, the fraction of a random
population that is non-dominated grows quickly with :math:`M`. Past three or
four objectives almost everything is rank 1, dominance stops discriminating,
and the entire selection pressure rests on crowding distance --- which is a
poor proxy for spread in high dimensions, since it measures each objective
independently rather than the true distance between points. This is the
*many-objective* problem, and it is why NSGA-III and RVEA exist.


NSGA-III
--------

:class:`~aeroopt.optimization.stochastic.nsgaiii.NSGAIII`

NSGA-III keeps non-dominated sorting but replaces crowding distance with
diversity enforced against a **fixed set of reference directions**, spread
evenly over the objective simplex.

Das-Dennis reference points
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~aeroopt.optimization.moea.DecompositionBasedAlgorithm.das_dennis_reference_points`
generates all vectors whose components are multiples of :math:`1/p` summing
to 1:

.. math::

   \lambda \in \left\{ \left(\tfrac{k_1}{p}, \dots, \tfrac{k_M}{p}\right)
   \ \middle|\ \sum_i k_i = p,\ k_i \in \mathbb{Z}_{\ge 0} \right\}

There are :math:`\binom{p + M - 1}{M - 1}` of them.
:meth:`~aeroopt.optimization.moea.DecompositionBasedAlgorithm.suggest_n_partitions`
picks :math:`p` so that count lands near ``population_size``.

Normalization
^^^^^^^^^^^^^

Before association, objectives are shifted by the ideal point and scaled by the
**intercepts** of the hyperplane through the extreme points, found via an
achievement scalarizing function. This adapts the reference grid to the actual
scale of the observed front, so objectives with different magnitudes still get
comparable representation. A singular extreme-point matrix (which happens when
the front is degenerate) falls back to the per-objective maxima.

Niching
^^^^^^^

Each individual is associated with its nearest reference direction by
perpendicular distance. When filling the last front, NSGA-III repeatedly:

1. finds the reference direction with the **fewest** associated survivors;
2. from the candidates associated with it, takes the one closest to the line;
3. increments that direction's niche count.

Because the least-represented direction is always served first, the survivors
spread across the whole reference set. Diversity becomes a property of the
predefined grid rather than something inferred from the population --- which is
exactly what makes it robust when dominance has stopped discriminating.

Reference: Deb & Jain (2014), *IEEE TEC* 18(4):577-601.


RVEA
----

:class:`~aeroopt.optimization.stochastic.rvea.RVEA`

RVEA drops the niche-counting bookkeeping. Individuals are partitioned by
**angle** to the nearest reference vector, and within each partition exactly one
survivor is chosen by the smallest **angle-penalized distance**:

.. math::

   \mathrm{APD}_{i} = \|f_i - z^*\| \cdot
   \left(1 + M \left(\frac{t}{T}\right)^{\alpha} \frac{\theta_{i,k}}{\gamma_k}\right)

with

* :math:`\|f_i - z^*\|` --- distance to the ideal point: **convergence**.
* :math:`\theta_{i,k}` --- angle between the individual and its reference
  vector: **diversity**.
* :math:`\gamma_k` --- the angle from vector :math:`k` to its nearest
  neighbouring vector, which normalizes the penalty for locally denser regions
  of the reference set.
* :math:`(t/T)^{\alpha}` --- the search progress, raised to ``alpha``.

The progress term is the interesting part. Early on it is near zero, the penalty
vanishes, and APD reduces to pure distance-to-ideal: the population is free to
race towards the front from any direction. As :math:`t \to T` the angular term
dominates and the population is pushed to spread along its assigned vectors.
**Convergence first, diversity second**, on a schedule instead of a fixed
trade-off.

Reference-vector adaptation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uniformly spread vectors only produce a uniformly spread front if the objectives
have comparable ranges. Every ``adapt_freq`` fraction of the run
(:class:`~aeroopt.optimization.stochastic.rvea.RVEAApdState`), the vectors are
rescaled by the observed ideal-to-nadir span and renormalized:

.. math::

   V_k \leftarrow \frac{V_k^{0} \odot (z^{\text{nad}} - z^{*})}
                       {\|V_k^{0} \odot (z^{\text{nad}} - z^{*})\|}

so the reference set stretches to match the front the run is actually finding.

.. warning::

   Because the APD penalty depends explicitly on :math:`t/T`, RVEA's behaviour
   is tied to ``max_iterations``. Stopping a run early does not give you "RVEA
   with fewer iterations" --- it gives you a run that never reached its
   diversity-preserving phase. Set ``max_iterations`` to the budget you actually
   intend to spend.

Reference: Cheng, Jin, Olhofer & Sendhoff (2016), *IEEE TEC* 20(5):773-791.


Choosing between them
---------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Situation
     - Suggestion
   * - 2-3 objectives, general use
     - **NSGA-II**. Mature, parameter-light, no reference grid to size.
   * - 4+ objectives
     - **NSGA-III** or **RVEA**; crowding distance is no longer informative.
   * - Objectives on very different scales
     - **RVEA**, whose vector adaptation targets exactly this.
   * - Front shape unknown or irregular
     - **NSGA-II** or **NSGA-III**; a fixed reference grid suits regular fronts best.
   * - Continuous variables, rugged landscape
     - **DE**, whose difference vectors adapt to the local landscape.
   * - Fixed, well-understood budget
     - **RVEA** benefits from a correctly set ``max_iterations``.
