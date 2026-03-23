import numpy as np
from scipy import special


class DasDennis:
    """Enumerates Das–Dennis reference directions (uniform simplex lattice) via a stack-based traversal."""

    def __init__(self, n_partitions, n_dim, scaling=None):
        """Set lattice fineness ``n_partitions``, simplex dimension ``n_dim``, and optional affine scaling of weights."""
        super().__init__()
        self.n_partitions = n_partitions
        self.n_dim = n_dim
        self.scaling = scaling

        self.stack = []
        self.stack.append(([], self.n_partitions))

    def number_of_points(self):
        """Return the binomial count of lattice points for the current ``n_dim`` and ``n_partitions``."""
        return int(special.binom(self.n_dim + self.n_partitions - 1, self.n_partitions))

    def next(self, n_points=None):
        """Yield up to ``n_points`` weight vectors (or all remaining) from the traversal."""
        ret = []
        self.traverse(lambda p: ret.append(p), n_points)
        return np.array(ret)

    def has_next(self):
        """True if the stack still holds unfinished lattice branches."""
        return len(self.stack) > 0

    def traverse(self, func, n_points=None):
        """Depth-first expansion: pop partial compositions, complete full simplex vertices, call ``func`` on each."""
        if self.n_partitions == 0:
            return np.full((1, self.n_dim), 1 / self.n_dim)

        counter = 0

        while (n_points is None or counter < n_points) and len(self.stack) > 0:

            point, beta = self.stack.pop()

            if len(point) + 1 == self.n_dim:
                point.append(beta / (1.0 * self.n_partitions))

                if self.scaling is not None:
                    point = [p * self.scaling + ((1 - self.scaling) / len(point)) for p in point]

                func(point)
                counter += 1
            else:
                for i in range(beta + 1):
                    _point = list(point)
                    _point.append(1.0 * i / (1.0 * self.n_partitions))
                    self.stack.append((_point, beta - i))

        return counter
