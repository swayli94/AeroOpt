Surrogate-driven optimization
=============================

The premise
-----------

An evolutionary algorithm typically needs thousands of evaluations. If one
evaluation is a 30-minute CFD run, that budget does not exist.

A surrogate model breaks the deadlock: fit a cheap statistical approximation to
the designs already evaluated, run the expensive *search* on the approximation,
and spend the real evaluations only on the handful of designs the approximation
recommends. The evaluation budget shifts from "how many designs can we try" to
"which designs are worth trying".


Kriging
-------

:class:`~aeroopt.utils.surrogate.Kriging` wraps ``smt``'s KPLS model. Kriging
(Gaussian process regression) is the standard choice for this role for one
reason: it returns not just a prediction but a **variance**, and that variance
is what makes intelligent sampling possible.

The model interpolates the training data exactly and, away from it, reverts
towards the trend with growing uncertainty. The result is a predicted mean
:math:`\hat{y}(x)` plus an epistemic standard deviation :math:`\sigma(x)` that
is zero at every sampled point and largest in unexplored regions.

KPLS adds a partial-least-squares projection so the correlation
hyperparameters are fitted in a reduced space, which keeps training tractable as
the number of design variables grows --- ordinary Kriging becomes impractical
well before 50 dimensions.

One independent single-output model is fitted per predicted output. Training on
scaled data (``train_on_scaled_data=True``, the default) is recommended: the
correlation lengths then live in a common :math:`[0, 1]` space and do not have
to absorb differences in physical units.


Exploration versus exploitation
-------------------------------

Optimizing the predicted mean alone is the classic failure mode: the search
converges onto whatever the model currently believes is best, adds samples
there, and never discovers that a better region exists somewhere it never
looked. The model becomes confidently wrong.

:meth:`~aeroopt.utils.surrogate.Kriging.predict_for_adaptive_sampling` avoids
this by optimizing a **confidence bound** instead of the mean --- shifting the
prediction by one standard deviation in the improving direction:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Output role
     - Criterion
     - Effect
   * - minimize (``-1``)
     - :math:`\hat{y} - \sigma` (LCB)
     - Prefers low predictions **and** uncertain regions.
   * - maximize (``1``)
     - :math:`\hat{y} + \sigma` (UCB)
     - Prefers high predictions **and** uncertain regions.
   * - other (``0``, ``2``)
     - :math:`\sigma`
     - Pure exploration of that output.

Because :math:`\sigma` collapses to zero wherever a real evaluation has been
made, the criterion automatically stops recommending places that have already
been sampled: the model self-corrects as data accumulates.


SBO: surrogate-based optimization
---------------------------------

:class:`~aeroopt.optimization.hybrid.sbo.SBO`

Every candidate comes from the surrogate. Each outer iteration:

1. **Retrain** the surrogate on all of ``db_valid``.
2. **Search** it: an inner optimizer (any ``Opt*`` driver) runs to completion on
   the confidence-bound criteria. This is free --- thousands of surrogate
   evaluations cost seconds.
3. **Truncate** the inner result to a non-dominated parent pool.
4. **Evaluate** those designs for real, and absorb them into the archive.

.. code-block:: text

   db_valid ──train──► surrogate ──inner optimization──► promising designs
       ▲                                                        │
       └──────────────── real evaluation ◄──────────────────────┘

This extracts the maximum value per real evaluation, and is the right choice
when evaluations are genuinely expensive relative to model training.

The risk is total dependence on model quality. If the surrogate is inaccurate,
every candidate in the iteration is wasted. :class:`~aeroopt.optimization.hybrid.sbo.PostProcessSBO`
exists to make that visible: it reports the prediction error on the candidates
that were just evaluated for real, which is honest out-of-sample validation.


SAO: surrogate-assisted optimization
------------------------------------

:class:`~aeroopt.optimization.hybrid.sao.SAO`

SAO hedges. Each iteration produces candidates from **both** sources:

* **"E" individuals** --- ordinary differential-evolution offspring of the real
  archive.
* **"S" individuals** --- designs found by optimizing the surrogate.

A fraction ``ratio_from_surrogate`` of the population is replaced by "S"
candidates; the rest stay "E". Progress therefore continues even while the
surrogate is still poor, because the evolutionary search does not depend on it.

:class:`~aeroopt.optimization.hybrid.sao.PostProcessSAO` reports both accuracy
and **contribution**: how many of each source landed on the candidates'
non-dominated front. That second number is the one that answers the practical
question --- is the surrogate actually earning the evaluations it is being
given? If "S" individuals never reach the front, lower
``ratio_from_surrogate``; if they dominate it, raise it.


Choosing between them
---------------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Situation
     - Suggestion
   * - Evaluations extremely expensive, model trusted
     - **SBO** --- maximum information per evaluation.
   * - Model quality unknown or unstable
     - **SAO** --- the evolutionary half keeps the run moving.
   * - High-dimensional design space
     - **SAO**; Kriging accuracy degrades with dimension.
   * - Very few initial samples
     - **SAO**, or SBO only after seeding a reasonable design of experiments.
   * - Noisy evaluations
     - Kriging with a noise term; exact interpolation of noise is harmful.


Using another surrogate backend
-------------------------------

:class:`~aeroopt.utils.surrogate.SurrogateModel` is an abstract interface, and
nothing in SBO or SAO depends on ``smt``. Implement five methods --- ``train``,
``predict``, ``full_predict``, ``predict_for_adaptive_sampling`` and
``evaluate_performance`` --- against any backend (scikit-learn, GPyTorch, a
neural network) and both frameworks work unchanged.

The only real requirement is that ``full_predict`` return an
``epistemic_variance``. A model that cannot express its own uncertainty can
still be used, but the adaptive-sampling criterion degenerates to the predicted
mean, with the failure mode described above.
