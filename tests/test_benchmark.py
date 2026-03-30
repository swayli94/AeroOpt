"""
Tests for aeroopt.utils.benchmark.
"""

import numpy as np
import pytest

from aeroopt.utils import benchmark as bm


# -----------------------------------------------------------------------------
# Scalar functions: known minima and return type
# -----------------------------------------------------------------------------


class TestSphere:
    def test_min_at_zero(self):
        x = np.zeros(4)
        assert bm.Sphere(x) == 0.0

    def test_value(self):
        x = np.array([1.0, 2.0, 3.0])
        assert bm.Sphere(x) == 1 + 4 + 9

    def test_returns_float(self):
        assert isinstance(bm.Sphere(np.array([1.0])), float)


class TestSumSphere:
    def test_min_at_zero(self):
        assert bm.SumSphere(np.zeros(3)) == 0.0

    def test_value(self):
        x = np.array([1.0, 1.0, 1.0])
        # 1*1 + 2*1 + 3*1 = 6
        assert bm.SumSphere(x) == 6.0


class TestPerm:
    def test_min_at_identity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert bm.Perm(x) == pytest.approx(0.0, abs=1e-10)

    def test_positive_off_min(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        assert bm.Perm(x) > 0


class TestPerm0:
    def test_min_at_inverse(self):
        x = 1.0 / np.arange(1, 5.0)
        assert bm.Perm0(x) == pytest.approx(0.0, abs=1e-10)

    def test_returns_float(self):
        assert isinstance(bm.Perm0(np.ones(3)), float)


class TestStyblinskiTang:
    def test_known_point(self):
        # approx minimizer -2.903534
        x = np.full(2, -2.903534)
        y = bm.StyblinskiTang(x)
        assert y == pytest.approx(2 * (-39.16617), rel=1e-3)

    def test_returns_float(self):
        assert isinstance(bm.StyblinskiTang(np.zeros(3)), float)


class TestDixonPrice:
    def test_min_at_known(self):
        # global min at x_i = 2^(-(2^i - 2) / 2^i), i=1..n
        n = 4
        x = np.array([2 ** (-(2**i - 2) / 2**i) for i in range(1, n + 1)])
        assert bm.DixonPrice(x) == pytest.approx(0.0, abs=1e-10)

    def test_1d(self):
        assert bm.DixonPrice(np.array([1.0])) == 0.0


class TestRosenbrock:
    def test_min_at_ones(self):
        x = np.ones(5)
        assert bm.Rosenbrock(x) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        x = np.array([1.0, 1.0])
        assert bm.Rosenbrock(x) == 0.0


class TestSchwefel:
    def test_known_optimum(self):
        # global min near 420.9687 per dim
        x = np.full(3, 420.9687)
        assert bm.Schwefel(x) == pytest.approx(0.0, abs=1e-2)


class TestRastrigin:
    def test_min_at_zero(self):
        x = np.zeros(4)
        assert bm.Rastrigin(x) == pytest.approx(0.0, abs=1e-10)

    def test_default_A(self):
        assert bm.Rastrigin(np.zeros(2)) == 0.0

    def test_custom_A(self):
        x = np.zeros(2)
        assert bm.Rastrigin(x, A=5.0) == 0.0


class TestGriewank:
    def test_min_at_zero(self):
        x = np.zeros(4)
        assert bm.Griewank(x) == pytest.approx(0.0, abs=1e-10)


class TestGaussian:
    def test_at_zero(self):
        assert bm.Gaussian(np.zeros(3)) == 1.0

    def test_negative_at_origin(self):
        x = np.array([0.5, 0.5])
        assert 0 < bm.Gaussian(x) < 1


class TestGussian:
    def test_alias_equals_Gaussian(self):
        x = np.array([0.1, 0.2])
        assert bm.Gussian(x) == bm.Gaussian(x)


class TestAckley:
    def test_min_at_zero(self):
        x = np.zeros(5)
        assert bm.Ackley(x) == pytest.approx(0.0, abs=1e-10)


# -----------------------------------------------------------------------------
# 2D-only scalar functions
# -----------------------------------------------------------------------------


class TestFranke:
    def test_2d_ok(self):
        x = np.array([0.5, 0.5])
        y = bm.Franke(x)
        assert isinstance(y, float)
        assert np.isfinite(y)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.Franke(np.array([0.5, 0.5, 0.5]))


class TestDroplet:
    def test_2d_ok(self):
        x = np.array([0.0, 0.0])
        y = bm.Droplet(x)
        assert isinstance(y, float)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.Droplet(np.ones(3))


class TestDeJong2nd:
    def test_min_at_1_1(self):
        x = np.array([1.0, 1.0])
        assert bm.DeJong2nd(x) == pytest.approx(0.0, abs=1e-10)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.DeJong2nd(np.ones(1))


# -----------------------------------------------------------------------------
# Multi-output (Pareto)
# -----------------------------------------------------------------------------


class TestCircle2D:
    def test_shape(self):
        x = np.array([0.5, 0.5])
        y = bm.Circle2D(x)
        assert y.shape == (2,)
        assert y.dtype == x.dtype

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.Circle2D(np.ones(3))


class TestConstr:
    def test_shape(self):
        x = np.array([0.5, 1.0])
        y = bm.Constr(x)
        assert y.shape == (2,)
        assert y[0] == 0.5
        assert y[1] == (1 + 1) / 0.5

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.Constr(np.ones(1))


class TestSrn:
    def test_shape_and_value(self):
        x = np.array([2.0, 1.0])
        y = bm.Srn(x)
        assert y.shape == (2,)
        assert y[0] == pytest.approx(2.0)
        assert y[1] == pytest.approx(18.0)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D problem"):
            bm.Srn(np.ones(3))


class TestZDT1:
    def test_shape(self):
        x = np.zeros(5)
        y = bm.ZDT1(x)
        assert y.shape == (2,)

    def test_f1_equals_x0(self):
        x = np.array([0.3, 0.5, 0.1])
        y = bm.ZDT1(x)
        assert y[0] == pytest.approx(0.3)

    def test_nx_1_raises(self):
        with pytest.raises(ValueError, match="nx >= 2"):
            bm.ZDT1(np.array([0.5]))


class TestZDT2:
    def test_shape(self):
        y = bm.ZDT2(np.array([0.2, 0.3, 0.4]))
        assert y.shape == (2,)

    def test_nx_1_raises(self):
        with pytest.raises(ValueError, match="nx >= 2"):
            bm.ZDT2(np.array([0.5]))


class TestZDT3:
    def test_shape(self):
        y = bm.ZDT3(np.array([0.1, 0.2, 0.3]))
        assert y.shape == (2,)

    def test_nx_1_raises(self):
        with pytest.raises(ValueError, match="nx >= 2"):
            bm.ZDT3(np.array([0.5]))


class TestZDT4:
    def test_shape(self):
        x = np.array([0.5] + [0.5] * 4)  # 5 dims
        y = bm.ZDT4(x)
        assert y.shape == (2,)

    def test_nx_1_raises(self):
        with pytest.raises(ValueError, match="nx >= 2"):
            bm.ZDT4(np.array([0.5]))


class TestZDT6:
    def test_shape(self):
        y = bm.ZDT6(np.array([0.25, 0.5, 0.25]))
        assert y.shape == (2,)

    def test_nx_1_raises(self):
        with pytest.raises(ValueError, match="nx >= 2"):
            bm.ZDT6(np.array([0.5]))


# -----------------------------------------------------------------------------
# Custom (Sway LI)
# -----------------------------------------------------------------------------


class TestRastriginSphere:
    def test_4d_shape(self):
        x = np.zeros(4)
        y = bm.RastriginSphere(x)
        assert y.shape == (2,)
        assert y[0] == 0.0
        assert y[1] == 0.0

    def test_y1_clipped(self):
        # large Rastrigin so y1 would be > 1, clipped to 1
        x = np.array([2.0, 2.0, 0.0, 0.0])
        y = bm.RastriginSphere(x)
        assert y[0] <= 1.0
        assert y[1] >= 0


class TestOneDimProblem:
    def test_1d_ok(self):
        x = np.array([10.0])
        y = bm.OneDimProblem(x)
        assert isinstance(y, float)
        assert np.isfinite(y)

    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            bm.OneDimProblem(np.array([1.0, 2.0]))


# -----------------------------------------------------------------------------
# Gradient
# -----------------------------------------------------------------------------


class TestGradient:
    def test_sphere_gradient_analytic(self):
        # grad(||x||^2) = 2*x
        x = np.array([1.0, 2.0, -0.5])
        g = bm.Gradient(x, bm.Sphere)
        np.testing.assert_allclose(g, 2 * x, rtol=1e-5)

    def test_rosenbrock_near_min(self):
        x = np.ones(3)
        g = bm.Gradient(x, bm.Rosenbrock)
        assert g.shape == (3,)
        np.testing.assert_allclose(g, np.zeros(3), atol=1e-4)

    def test_vector_output_raises(self):
        with pytest.raises(TypeError, match="scalar-output"):
            bm.Gradient(np.array([0.5, 0.5]), bm.ZDT1)

    def test_returns_same_dtype(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        g = bm.Gradient(x, bm.Sphere)
        assert g.dtype == x.dtype
        assert g.shape == (2,)


# -----------------------------------------------------------------------------
# Batch / edge cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    def test_sphere_1d(self):
        assert bm.Sphere(np.array([3.0])) == 9.0

    def test_rastrigin_1d(self):
        assert bm.Rastrigin(np.zeros(1)) == 0.0

    def test_ackley_1d(self):
        assert bm.Ackley(np.zeros(1)) == pytest.approx(0.0, abs=1e-10)
