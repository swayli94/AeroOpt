'''
Example: Kriging surrogate model for deterministic, homoscedastic, heteroscedastic functions.

- Test functions (n_input = 1, n_output = 1, x in [0, 1]):
  1) deterministic: f(x) = sin(3*pi*x)
  2) homoscedastic: f(x) = sin(3*pi*x) + 0.1*epsilon, epsilon ~ N(0, 1)
  3) heteroscedastic: f(x) = sin(3*pi*x) + 0.1*x*epsilon, epsilon ~ N(0, 1)

- Data generation:
  1) generate N_A data points (x, y), denoted by setA (x ~ Uniform(0, 1) i.i.d.)
  2) sample N_B points from setA (with replacement on indices), redraw y from the
     stochastic model, denoted by setB (duplicated x are used by SMT het noise)

- Use `smt.surrogate_models.KPLS` as the surrogate model.
  1) https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/gpr/kpls.html
  2) train deterministic KPLS on setA (function 1)
  3) train stochastic KPLS (eval_noise, homoscedastic) on setA + setB (function 2)
  4) train heteroscedastic KPLS (eval_noise + use_het_noise) on setA + setB (function 3)

- Plot results:
  1) each function in a separate subplot
  2) plot the true mean (and true 1-sigma band when noise is present)
  3) plot the data points
  4) plot the surrogate mean, 1-sigma epistemic band (from predict_variances),
     and 1-sigma total predictive band (epistemic + learned aleatoric from optimal_noise)

'''

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import get_backend
import numpy as np
from smt.surrogate_models import KPLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
NOISE_SCALE = 0.4
N_GRID = 400
N_A = 10
N_B = 10

# Duplicate x (set B reuses x from A) needs a non-trivial noise prior so the GP
# covariance stays numerically PD; see smt.surrogate_models.krg_based defaults.
_KPLS_NOISE_COMMON = dict(
    print_global=False,
    hyper_opt="Cobyla",
    nugget=1e-8,
    noise0=[1e-3],
    n_start=15,
)


def mean_sin(x: np.ndarray) -> np.ndarray:
    return np.sin(6.0 * np.pi * x) * x * np.exp(x)


def build_sets(
    rng: np.random.Generator,
    noise: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (x_train, y_train) for setA only, then full training (setA + setB).

    noise: 'none' | 'homo' | 'hetero'
    """
    n_a = N_A
    x_a = rng.uniform(0.0, 1.0, (n_a, 1))

    if noise == "none":
        y_a = mean_sin(x_a)
    elif noise == "homo":
        eps = rng.standard_normal((n_a, 1))
        y_a = mean_sin(x_a) + NOISE_SCALE * eps
    elif noise == "hetero":
        eps = rng.standard_normal((n_a, 1))
        y_a = mean_sin(x_a) + NOISE_SCALE * x_a * eps
    else:
        raise ValueError(noise)

    idx_b = rng.integers(0, n_a, size=N_B)
    x_b = x_a[idx_b]
    if noise == "none":
        y_b = mean_sin(x_b)
    elif noise == "homo":
        eps_b = rng.standard_normal((len(x_b), 1))
        y_b = mean_sin(x_b) + NOISE_SCALE * eps_b
    else:
        eps_b = rng.standard_normal((len(x_b), 1))
        y_b = mean_sin(x_b) + NOISE_SCALE * x_b * eps_b

    x_all = np.vstack([x_a, x_b])
    y_all = np.vstack([y_a, y_b])
    return x_a, y_a, x_all, y_all


def kpls_aleatoric_variance(sm: KPLS, x: np.ndarray) -> np.ndarray:
    """
    Observation noise variance at prediction sites (physical y units).

    SMT ``predict_variances`` is epistemic (latent GP); with ``eval_noise`` the
    aleatoric part is ``optimal_noise * sigma2`` (homoscedastic scalar noise ratio,
    heteroscedastic ratios interpolated over unique training sites).
    """
    if not sm.options["eval_noise"]:
        return np.zeros(x.shape[0], dtype=float)
    sigma2 = float(np.asarray(sm.optimal_par["sigma2"], dtype=float).ravel()[0])
    nu = np.asarray(sm.optimal_noise, dtype=float)
    if not sm.options["use_het_noise"]:
        nu_s = float(nu.ravel()[0])
        return np.full(x.shape[0], nu_s * sigma2, dtype=float)
    xu = np.asarray(sm.X_norma, dtype=float).ravel()
    nu_v = nu.ravel()
    order = np.argsort(xu)
    xq = ((x - sm.X_offset) / sm.X_scale).ravel()
    nu_i = np.interp(xq, xu[order], nu_v[order])
    return nu_i * sigma2


def true_one_sigma_band(
    x: np.ndarray, noise: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and +/- 1 sigma of the generative process (aleatoric)."""
    m = mean_sin(x).ravel()
    if noise == "none":
        lo = hi = m
    elif noise == "homo":
        s = np.full_like(m, NOISE_SCALE)
        lo, hi = m - s, m + s
    else:
        s = (NOISE_SCALE * x.ravel())
        lo, hi = m - s, m + s
    return m, lo, hi


def subplot_panel(
    ax: plt.Axes,
    title: str,
    noise: str,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sm: KPLS,
    x_fine: np.ndarray,
    ) -> None:
    m_true, lo_true, hi_true = true_one_sigma_band(x_fine, noise)

    ax.plot(x_fine.ravel(), m_true, "k-", label="true mean", linewidth=1.2)
    if noise != "none":
        ax.fill_between(
            x_fine.ravel(),
            lo_true,
            hi_true,
            color="k",
            alpha=0.2,
            label="true ±1σ (aleatoric)",
        )

    ax.scatter(x_a.ravel(), y_a.ravel(), s=22, c="C0", label="set A", zorder=4)
    if x_train.shape[0] > x_a.shape[0]:
        # hide duplicate scatter for points already in A for clarity: only extra B rows
        # (all B indices are from A so we mark second-half as set B)
        n_extra = x_train.shape[0] - x_a.shape[0]
        if n_extra > 0:
            ax.scatter(
                x_train[-n_extra:].ravel(),
                y_train[-n_extra:].ravel(),
                s=22,
                facecolors="none",
                edgecolors="C1",
                linewidths=1.2,
                label="set B",
                zorder=5,
            )

    y_hat = sm.predict_values(x_fine)
    v_epi = np.asarray(sm.predict_variances(x_fine), dtype=float).ravel()
    v_epi = np.maximum(v_epi, 0.0)
    s_epi = np.sqrt(v_epi)
    y_hat = np.asarray(y_hat, dtype=float).ravel()

    v_ale = kpls_aleatoric_variance(sm, x_fine)
    v_ale = np.maximum(v_ale, 0.0)
    s_ale = np.sqrt(v_ale)

    ax.plot(x_fine.ravel(), y_hat, "C3--", label="KPLS mean", linewidth=1.2)
    xf = x_fine.ravel()
    ax.fill_between(
        xf,
        y_hat - s_epi,
        y_hat + s_epi,
        color="C2",
        alpha=0.1,
        label="KPLS ±1σ (epistemic)",
    )
    if np.any(v_ale > 0.0):
        ax.fill_between(
            xf,
            y_hat - s_ale,
            y_hat + s_ale,
            color="C3",
            alpha=0.4,
            label="KPLS ±1σ (aleatoric)",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Warning: multiple x input features have the same value.*",
        category=UserWarning,
    )

    rng = np.random.default_rng(SEED)
    x_fine = np.linspace(0.0, 1.0, N_GRID).reshape(-1, 1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)

    # 1) Deterministic: train on set A only
    x_a, y_a, _, _ = build_sets(rng, "none")
    sm1 = KPLS(theta0=[1e-2], n_comp=1, print_global=False, hyper_opt="Cobyla")
    sm1.set_training_values(x_a, y_a)
    sm1.train()
    subplot_panel(
        axes[0],
        "Deterministic (noise-free) — KPLS on set A",
        "none",
        x_a,
        y_a,
        x_a,
        y_a,
        sm1,
        x_fine,
    )

    # 2) Homoscedastic: eval_noise, set A + B
    rng2 = np.random.default_rng(SEED)
    x_a2, y_a2, x_all2, y_all2 = build_sets(rng2, "homo")
    sm2 = KPLS(
        theta0=[1e-2],
        n_comp=1,
        eval_noise=True,
        use_het_noise=False,
        **_KPLS_NOISE_COMMON,
    )
    sm2.set_training_values(x_all2, y_all2)
    sm2.train()
    subplot_panel(
        axes[1],
        "Homoscedastic noise — KPLS with eval_noise (A + B)",
        "homo",
        x_a2,
        y_a2,
        x_all2,
        y_all2,
        sm2,
        x_fine,
    )

    # 3) Heteroscedastic: replicate x in A ∪ B; SMT aggregates & estimates local noise
    rng3 = np.random.default_rng(SEED)
    x_a3, y_a3, x_all3, y_all3 = build_sets(rng3, "hetero")
    sm3 = KPLS(
        theta0=[1e-2],
        n_comp=1,
        eval_noise=True,
        use_het_noise=True,
        **_KPLS_NOISE_COMMON,
    )
    sm3.set_training_values(x_all3, y_all3)
    sm3.train()
    subplot_panel(
        axes[2],
        "Heteroscedastic noise — KPLS with eval_noise + use_het_noise (A + B)",
        "hetero",
        x_a3,
        y_a3,
        x_all3,
        y_all3,
        sm3,
        x_fine,
    )

    out_png = Path(__file__).resolve().parent / "example_kriging.png"
    fig.savefig(out_png, dpi=150)
    if "agg" not in get_backend().lower():
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
