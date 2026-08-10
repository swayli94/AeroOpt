Decomposition-based algorithms
==============================

MOEA/D
------

:class:`~aeroopt.optimization.stochastic.moead.MOEAD`

The dominance-based algorithms treat the population as one set and impose
diversity on it. MOEA/D changes the question entirely: it converts the
multi-objective problem into :math:`N` **single-objective subproblems**, each
defined by a weight vector, and solves them cooperatively.

Every weight vector :math:`\lambda_k` owns one solution slot. Diversity is not
maintained --- it is structural, because the weights are spread evenly over the
simplex by construction.

.. important::

   This is why ``population_size`` **must** equal the number of Das-Dennis
   reference points: one slot per weight vector, no more and no less. For *M*
   objectives and *p* partitions that is :math:`\binom{p+M-1}{M-1}`, i.e.
   :math:`p+1` for two objectives.
   :class:`~aeroopt.optimization.stochastic.moead.OptMOEAD` checks this at
   construction and raises rather than failing obscurely later.


Scalarization
-------------

:meth:`~aeroopt.optimization.moea.DecompositionBasedAlgorithm.decomposed_values`
implements two ways of collapsing an objective vector onto a scalar, given the
ideal point :math:`z^*`. Smaller is better in both.

Tchebycheff
^^^^^^^^^^^

.. math::

   g^{\text{te}}(f \mid \lambda, z^*) =
   \max_{i} \ \lambda_i \left| f_i - z^*_i \right|

Only the worst weighted deviation counts, which forces balanced improvement
across objectives. Its decisive property is that it can reach solutions on
**non-convex** parts of the front --- unlike a weighted sum, which can only ever
find the convex hull. Preferred for two objectives.

Penalty-based boundary intersection (PBI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PBI splits the displacement from the ideal point into a component along the
reference direction and one perpendicular to it:

.. math::

   d_1 = \frac{(f - z^*)^\top \lambda}{\|\lambda\|},
   \qquad
   d_2 = \left\| (f - z^*) - d_1 \frac{\lambda}{\|\lambda\|} \right\|

.. math::

   g^{\text{pbi}}(f \mid \lambda, z^*) = d_1 + \theta\, d_2

:math:`d_1` measures **convergence** (how far along the ray) and :math:`d_2`
measures **diversity** (how far off it). ``pbi_theta`` sets the exchange rate;
5.0 is the usual value. Small :math:`\theta` lets solutions drift off their
direction, large :math:`\theta` pins them to it at the cost of convergence
speed.

PBI produces more uniform fronts than Tchebycheff in higher dimensions, which is
why ``decomposition: "auto"`` selects Tchebycheff for :math:`M \le 2` and PBI
otherwise.


Neighbourhoods
--------------

Subproblems with similar weight vectors have similar optima, so their solutions
are useful to each other. Each weight's :math:`T` nearest weights (Euclidean
distance in weight space,
:meth:`~aeroopt.optimization.stochastic.moead.MOEAD.neighbor_indices`) form its
neighbourhood, and that neighbourhood is used twice:

* **Mating.** With probability ``prob_neighbor_mating`` both parents are drawn
  from the neighbourhood; otherwise from the whole population. The occasional
  global draw is what keeps the search from fragmenting into independent local
  runs.
* **Replacement.** A new solution is offered to every neighbouring subproblem
  and takes over each slot whose scalarized value it improves. One good design
  can therefore upgrade several subproblems at once.

``n_neighbors`` trades locality against information flow: too small and
subproblems stop helping each other, too large and MOEA/D degenerates towards a
single global population.


The ideal point
---------------

:math:`z^*` is the component-wise best value seen so far, updated as evaluations
arrive (:meth:`~aeroopt.optimization.stochastic.moead.MOEAD.update_ideal`). Both
scalarizations measure from it, so it acts as the moving origin of the search.


Generational adaptation
-----------------------

Textbook MOEA/D is sequential: generate one offspring, evaluate it immediately,
replace neighbours, move to the next subproblem. That is incompatible with
expensive parallel evaluation, where the whole point is to submit a full batch.

:class:`~aeroopt.optimization.stochastic.moead.OptMOEAD` therefore generates one
offspring per subproblem in random order, queues the ``(subproblem, offspring)``
pairs, evaluates the entire batch at once, and applies all replacements
afterwards in generation order. The queue is flushed at the start of the next
``generate_candidate_individuals`` and again after the loop ends, so no
replacement is lost at the final generation.

The trade-off is that within one generation a subproblem cannot benefit from an
offspring created for its neighbour in the same batch. In exchange, a generation
costs one parallel evaluation round instead of :math:`N` sequential ones.

If fewer feasible individuals exist than there are weights, the slots are filled
round-robin from what is available: several subproblems temporarily share one
design, and neighbourhood replacement diversifies them as real solutions arrive.

Reference: Zhang & Li (2007), *IEEE TEC* 11(6):712-731.


Post-hoc analysis: lagging directions
-------------------------------------

The same scalarization machinery is useful for diagnosis after a run.
:meth:`~aeroopt.optimization.moea.DecompositionBasedAlgorithm.find_slow_directions`
computes, for each reference direction,

.. math::

   \min_i \ g(f_i \mid \lambda_j, z^*)

over the non-dominated set and returns the directions ordered worst-first. A
large best-achievable value means no design approaches the ideal point along
that preference direction --- a genuine gap in the trade-off surface, or a
region the search never reached.

.. note::

   This is an analysis tool, not part of any algorithm. Standard NSGA-III, RVEA
   and MOEA/D do **not** identify lagging directions and steer selection towards
   them; they use reference directions only for survival and mating structure.
   The example ``example/5-evolutionary-algorithm/example_pareto_analysis.py``
   demonstrates the diagnostic.
