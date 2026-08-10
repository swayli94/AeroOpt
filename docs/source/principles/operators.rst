Variation operators
===================

Once parents are selected, an algorithm has to turn them into new designs.
AeroOpt implements the two operator families that dominate real-valued
evolutionary optimization.

Simulated binary crossover (SBX)
--------------------------------

Classical genetic algorithms crossed *bit strings*. SBX
(:func:`~aeroopt.optimization.utils.sbx_crossover`) reproduces the statistical
behaviour of one-point binary crossover directly on real numbers: children are
distributed around their parents, with most of the probability mass near them
and a tail that reaches further out.

For a pair of parent values :math:`p_1 < p_2`, a spread factor
:math:`\beta_q` is drawn from

.. math::

   \beta_q = \begin{cases}
       (2u)^{\frac{1}{\eta_c + 1}}
           & u \le 0.5 \\[6pt]
       \left(\dfrac{1}{2(1-u)}\right)^{\frac{1}{\eta_c + 1}}
           & u > 0.5
   \end{cases}
   \qquad u \sim \mathcal{U}(0, 1)

and the children are placed symmetrically about the parents' midpoint:

.. math::

   c_{1,2} = \tfrac{1}{2}\big[(p_1 + p_2) \mp \beta_q (p_2 - p_1)\big]

The distribution index :math:`\eta_c` (``pow_sbx``) controls the spread:

* **Large** :math:`\eta_c` (say 20) --- children hug their parents. Exploitation.
* **Small** :math:`\eta_c` (say 2) --- children scatter widely. Exploration.

The implementation uses the bounded variant: :math:`\beta` is derived from each
parent's distance to the variable bounds, so children stay inside the design
space by construction instead of being clipped afterwards. Variables whose
parents differ by less than ``input_precision`` are skipped --- crossing values
that are equal at the achievable resolution only wastes randomness.

.. admonition:: Self-adaptive step size
   :class: note

   SBX has a property that matters late in a run: because the child spread is
   proportional to :math:`p_2 - p_1`, a converged population automatically takes
   small steps. The operator anneals itself without any explicit schedule.


Polynomial mutation
-------------------

Crossover can only recombine values that already exist in the population. If
every parent has :math:`x_3 = 0.7`, no amount of crossover will produce
anything else. Mutation (:func:`~aeroopt.optimization.utils.polynomial_mutation`)
is what keeps that from becoming permanent.

Each variable is perturbed with probability ``mut_rate`` by

.. math::

   x' = x + \bar{\delta}\,(x^{\text{upp}} - x^{\text{low}})

where the perturbation :math:`\bar{\delta}` follows a polynomial distribution
whose index :math:`\eta_m` (``pow_poly``) plays the same role as
:math:`\eta_c`: larger means smaller typical steps. The distribution is also
bounded, so it shrinks near the edges of the design space and a mutated design
cannot leave it.

The configured ``mut_rate`` is interpreted as the **expected number of mutated
variables per individual**, and the drivers divide it by ``n_input`` before
calling the operator
(:attr:`~aeroopt.optimization.base.OptGeneticFramework.mut_rate_per_variable`).
So ``mut_rate = 1.0`` means "about one variable per child", independently of
whether the problem has 3 or 300 variables --- the setting keeps its meaning as
the problem grows.


Binary tournament selection
---------------------------

:func:`~aeroopt.optimization.utils.binary_tournament_selection` picks two
individuals at random and keeps the better one, repeated until the mating pool
is full.

"Better" is ``Individual.__lt__``, which compares by constraint violation first,
then Pareto rank, then crowding distance, then crowding potential. The
randomness is deliberate: a weaker design still wins whenever its opponent is
weaker still, which preserves the genetic material that a strictly greedy
selection would discard.


Differential evolution
----------------------

DE takes a different route: instead of a parametric distribution, it uses the
population's own spread as its step size.

For each target :math:`x_i`, three distinct other members are drawn
(:func:`~aeroopt.optimization.utils.sample_de_rand_1_indices`) and combined into
a mutant:

.. math::

   v_i = x_{r_0} + F \cdot (x_{r_1} - x_{r_2})

The difference vector :math:`x_{r_1} - x_{r_2}` is a sample of the population's
current distribution, so DE adapts its step size *and* its preferred direction
to the local landscape --- without estimating a covariance matrix. This is why
DE handles ill-conditioned and rotated problems well.

``scale_factor`` :math:`F` scales that difference; 0.5 is the usual starting
point.

Binomial crossover (:func:`~aeroopt.optimization.utils.binomial_crossover`) then
mixes mutant and target per variable, with one randomly chosen index always
taken from the mutant so the trial is never an exact copy of its target:

.. math::

   u_{i,j} = \begin{cases}
       v_{i,j} & \text{if } \mathrm{rand}_j \le CR \text{ or } j = j_{\text{rand}}\\
       x_{i,j} & \text{otherwise}
   \end{cases}

.. note::

   Classical DE compares each trial against its own target and keeps the winner.
   In AeroOpt every trial is added to the archive instead, and survival is
   decided by the shared rank-and-crowding environmental selection. This makes
   DE a multi-objective operator (a MODE-style algorithm) and keeps it
   consistent with the archive-based workflow.


Newton-Raphson-based optimizer
------------------------------

NRBO (:class:`~aeroopt.optimization.stochastic.nrbo.NRBO`, single-objective
only) borrows the root-finding step of Newton-Raphson to *estimate* a descent
direction from three population members --- best, worst and current --- without
ever computing a derivative:

.. math::

   \text{NRSR} \sim \mathcal{N}(0,1) \cdot
   \frac{(x_{\text{worst}} - x_{\text{best}})\, \Delta x}
        {2\,(x_{\text{worst}} + x_{\text{best}} - 2 x_{\text{now}})}

The denominator is a finite-difference stand-in for a second derivative, so the
step is large on flat stretches and small in sharp valleys. All divisions go
through a guarded helper, because that denominator vanishes whenever the three
points are collinear.

A **trap avoidance operator** fires with probability ``deciding_factor`` and
adds a randomized jump, either around the current individual or around the best
one. Its magnitude is scaled by

.. math::

   \delta = \left(1 - \frac{2t}{T}\right)^{5}

which decays sharply with iteration :math:`t`, so NRBO explores early and
refines late.

Reference: Sowmya, Premkumar & Jangir (2024),
*Engineering Applications of Artificial Intelligence* 128:107532.
