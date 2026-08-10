Crowding in the design space
============================

Two different notions of crowding
---------------------------------

NSGA-II's crowding distance measures spacing in the **objective** space: it asks
whether the trade-off surface is evenly covered.

:class:`~aeroopt.analysis.analyze_database.AnalyzeDatabase` measures crowding in
the **input** space instead: it asks whether the *design* space has been evenly
explored. These answer different questions, and both matter.

A study can have a beautifully spread Pareto front that came entirely from one
corner of the design space --- leaving whole regions unsampled and a surrogate
built on that data confidently wrong everywhere else. Input-space crowding is
what detects that, which is why it drives adaptive sampling and candidate
repair rather than survival selection.


The potential field
-------------------

Each evaluated design is treated as a source of a repulsive potential, and a new
candidate's "crowdedness" is the total potential it sits in.

The kernel (:func:`~aeroopt.analysis.utils.func_potential`) is

.. math::

   \phi(r) = (c\,r + 1)\, e^{-c r}

which has the properties this needs:

* :math:`\phi(0) = 1` --- a design coinciding with an existing one is maximally
  crowded.
* :math:`\phi \to 0` as :math:`r \to \infty`, and smoothly: distant designs
  contribute nothing without an arbitrary cut-off radius.
* :math:`\phi'(0) = 0` --- the field is flat at the source, so the metric is not
  hypersensitive to tiny displacements.

The total potential at a point is the sum over all designs in the database, so
it grows with both proximity *and* the number of nearby designs. A point wedged
between five neighbours scores worse than one with a single close neighbour.


Calibrating the length scale
----------------------------

The coefficient :math:`c` sets what counts as "close", and hard-coding it would
be meaningless: the right value depends on how densely the space has been
sampled, which changes as the study progresses.

It is derived instead. The **typical distance** :math:`d_{\text{typical}}` is
the mean nearest-neighbour distance over the database --- the current sampling
resolution. Then :math:`c` is solved from

.. math::

   \phi(d_{\text{typical}}) = \phi_{\text{crit}}

where :math:`\phi_{\text{crit}}` is ``critical_potential_x`` (default 0.2).
:func:`~aeroopt.analysis.utils.calculate_potential_coefficient` brackets and
bisects for :math:`c`, since the equation has no closed form.

The consequence is that potential values stay comparable across iterations: as
the database fills in and :math:`d_{\text{typical}}` shrinks, :math:`c` grows
and the field re-normalizes itself to the new resolution. A potential of 0.8
means the same thing at iteration 2 and at iteration 200.

Lowering ``critical_potential_x`` makes the field decay faster, so only very
close neighbours register.


What it is used for
-------------------

Two individual-level metrics are assigned by
:meth:`~aeroopt.analysis.analyze_database.AnalyzeDatabase.calculate_crowding_metrics`:

* ``crowding_distance`` --- distance to the nearest neighbour. **Higher is
  better** (more isolated).
* ``crowding_potential`` --- summed potential from all other designs. **Lower is
  better** (less crowded).

They support three tasks:

**Thinning an archive.**
:meth:`~aeroopt.analysis.analyze_database.AnalyzeDatabase.eliminate_crowding_individuals`
removes designs that are both too close to a neighbour and sitting in too much
potential, one at a time, recomputing the metrics after each removal. Removing
one member of a cluster changes how crowded the rest are, so a single-pass
threshold would over-prune. Useful for keeping a surrogate's training set
informative rather than merely large.

**Repairing candidates.**
:meth:`~aeroopt.optimization.base.PreProcess._restrict_x_values_by_valid_database`
pulls candidates into a distance band around known-good designs. Too close and
the evaluation is redundant; too far and a mesh generator or solver may fail to
converge. A candidate that duplicates an existing design is moved along the line
towards a randomly chosen nearby valid design, which keeps it in a region known
to work.

**Guiding sampling.**
:meth:`~aeroopt.analysis.analyze_database.AnalyzeDatabase.calculate_potential_induced_by_database`
scores arbitrary points, so a low-potential region is a candidate for the next
sample.


Grouping
--------

:meth:`~aeroopt.analysis.analyze_database.AnalyzeDatabase.calculate_grouping`
clusters designs with k-means and reports per-group statistics. On a
multi-modal problem this separates distinct design families --- several
qualitatively different shapes achieving comparable performance --- which a
single set of population-wide statistics would blur together.

The default ``random_state=0`` makes the grouping reproducible; pass ``None``
for scikit-learn's default behaviour.
