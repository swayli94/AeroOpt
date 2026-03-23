import numpy as np

from AeroOpt.utils.pymoo.misc import vectorized_cdist
from AeroOpt.utils.pymoo.ref_dirs.misc import project_onto_sum_equals_zero_plane, project_onto_unit_simplex_recursive
from AeroOpt.utils.pymoo.ref_dirs.optimizer import Adam
from AeroOpt.utils.pymoo.reference_direction import ReferenceDirectionFactory, map_onto_unit_simplex


def calc_dist_to_others(x, X):
    """Negated minimum Euclidean distance from x to each row of X (spread objective)."""
    return - np.sqrt(((x - X) ** 2).sum(axis=1)).min()


def calc_dist_to_others_with_gradient(x, X):
    """Value and gradient of ``calc_dist_to_others`` w.r.t. ``x`` (nearest neighbor drives the gradient)."""
    diff = (x - X)
    D = np.sqrt((diff ** 2).sum(axis=1))

    k = D.argmin()

    obj = - D.min()
    grad = - diff[k] / D[k]

    return obj, grad


class ConstructionBasedReferenceDirectionFactory(ReferenceDirectionFactory):
    """Greedily adds unit-simplex directions by maximizing distance to existing points, optionally refined by Adam."""

    def __init__(self,
                 n_dim,
                 n_points,
                 n_samples=100,
                 gradient_descent=True,
                 verbose=False,
                 **kwargs):
        """Configure objective dimension, target count, candidate pool size, and optional gradient refinement."""

        super().__init__(n_dim, **kwargs)
        self.n_points = n_points
        self.gradient_descent = gradient_descent
        self.n_samples = n_samples
        self.verbose = verbose
        self.X = None

    def _do(self, random_state=None, **kwargs):
        """Build ``n_points`` reference directions on the unit simplex and return them as rows."""

        self.random_state = random_state
        self.X = np.eye(self.n_dim)

        while len(self.X) < self.n_points:
            x = self.next()
            self.X = np.vstack([self.X, x])

            if self.verbose:
                print(len(self.X), "x", x)

        return self.X

    def next(self):
        """Pick the farthest candidate from ``self.X`` on the simplex, then optionally optimize spread with Adam."""

        x = self.random_state.random((self.n_samples, self.n_dim))
        x = map_onto_unit_simplex(x, "kraemer")
        x = x[vectorized_cdist(x, self.X).min(axis=1).argmax()]

        if self.gradient_descent:

            optimizer = Adam(precision=1e-4)

            # for each iteration of gradient descent
            for i in range(1000):

                # calculate the function value and the gradient
                # auto_obj, auto_grad = value_and_grad(calc_dist_to_others)(x, self.X)
                _obj, _grad = calc_dist_to_others_with_gradient(x, self.X)

                # project the gradient to have a sum of zero - guarantees to stay on the simplex
                proj_grad = project_onto_sum_equals_zero_plane(_grad)

                # apply a step of gradient descent by subtracting the projected gradient with a learning rate
                x = optimizer.next(x, proj_grad)

                # project the out of bounds points back onto the unit simplex
                project_onto_unit_simplex_recursive(x[None, :])

                # because of floating point issues make sure it is on the unit simplex
                x /= x.sum()

                # if there was only a little movement during the last iteration -> terminate
                if optimizer.has_converged:
                    break

        return x
