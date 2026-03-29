'''
Example: Pareto front analysis — slow Das-Dennis reference directions.

Builds a deliberately biased synthetic 2-objective front, wraps objectives in a
`Database`, runs `DecompositionBasedAlgorithm.find_slow_directions`, and plots:

  - objective space with rays colored by per-direction best Tchebycheff value;
  - a bar chart of ``best_achievement`` per direction.

Output: `slow_reference_directions_analysis.png` in this directory.
'''

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AeroOpt.core import Database, Individual, Problem
from AeroOpt.core.settings import SettingsData, SettingsProblem
from AeroOpt.optimization.moea import DecompositionBasedAlgorithm
from AeroOpt.optimization.utils import reference_directions


def _biased_pareto_for_slow_direction_demo(
        n: int, rng: np.random.Generator,
        ) -> np.ndarray:
    '''
    Mostly dense near small f1 (large f2): missing compromise region and f2-good
    corner so several Das-Dennis directions become comparatively ``slow''.
    '''
    n_left = int(round(0.78 * n))
    n_right = max(0, n - n_left)
    t_lo = np.sort(rng.uniform(0.06, 0.38, size=max(n_left, 1)))
    if n_left == 0:
        t_lo = np.array([], dtype=float)
    elif n_left < t_lo.size:
        t_lo = t_lo[:n_left]
    if n_right <= 0:
        t = t_lo
    else:
        t_hi = np.sort(rng.uniform(0.55, 0.95, size=n_right))
        t = np.sort(np.concatenate([t_lo, t_hi]))
    t = np.sort(t)
    f1 = t
    f2 = (1.0 - t) ** 2 + 0.025 + 0.018 * rng.standard_normal(size=len(t))
    f2 = np.clip(f2, 0.02, None)
    return np.stack([f1, f2], axis=1)


def _minimal_biobj_problem() -> Problem:
    '''Two minimization objectives, no constraints (for analysis demo only).'''
    cfg = {
        '_sd': {
            'type': 'SettingsData',
            'name': 'biobj',
            'name_input': ['x1', 'x2'],
            'name_output': ['f1', 'f2'],
            'input_low': [0.0, 0.0],
            'input_upp': [1.0, 1.0],
            'input_precision': [1.0e-6, 1.0e-6],
            # Match the ~[0, 1] synthetic objectives so `scale_y` is not (y+1e6)/2e6,
            # which squeezes all values near 0.5 and shrinks Tchebycheff gaps to ~1e-7.
            'output_low': [0.0, 0.0],
            'output_upp': [1.0, 1.0],
            'output_precision': [0.0, 0.0],
            'critical_scaled_distance': 1.0e-6,
        },
        '_sp': {
            'type': 'SettingsProblem',
            'name': 'biobj',
            'name_data_settings': 'biobj',
            'output_type': [1, 1],
            'constraint_strings': [],
        },
    }
    fd, path = tempfile.mkstemp(suffix='.json', prefix='aeroopt_biobj_')
    os.close(fd)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        sd = SettingsData('biobj', fname_settings=path)
        sp = SettingsProblem('biobj', sd, fname_settings=path)
        return Problem(sd, sp)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _database_from_f(problem: Problem, f: np.ndarray) -> Database:
    db = Database(problem, database_type='population')
    x0 = np.zeros(problem.n_input, dtype=float)
    for i in range(f.shape[0]):
        ind = Individual(
            problem, x=x0.copy(), ID=i + 1,
            y=np.asarray(f[i], dtype=float).reshape(-1))
        db.add_individual(
            ind, check_duplication=False, check_bounds=False,
            deepcopy=False, print_warning_info=False)
    return db


def main() -> None:
    '''
    Visualize `DecompositionBasedAlgorithm.find_slow_directions` / Tchebycheff
    best achievement per Das-Dennis direction on a deliberately biased front.
    '''
    rng = np.random.default_rng(2026)
    n_partitions = 6
    f = _biased_pareto_for_slow_direction_demo(110, rng)
    problem = _minimal_biobj_problem()
    db = _database_from_f(problem, f)

    ordered, best_g, ref = DecompositionBasedAlgorithm.find_slow_directions(
        db, n_partitions, pareto_front_only=True, decomposition='tchebicheff')

    ys = db.get_unified_objectives(scale=True)
    mask = DecompositionBasedAlgorithm._pareto_first_front_mask(ys)
    front = ys[mask]
    z_star = front.min(axis=0)
    ref_unit = reference_directions(ref)
    ray_len = float((front.max(axis=0) - z_star).max() * 1.2)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.15, 1.0], wspace=0.2)
    fig.set_constrained_layout_pads(w_pad=0.04)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    gmax = float(np.nanmax(best_g))
    gmin = float(np.nanmin(best_g))
    gn = Normalize(vmin=gmin, vmax=gmax + 1.0e-12, clip=True)
    cmap = colormaps['YlOrRd']

    ax0.scatter(
        front[:, 0], front[:, 1], c='#4c72b0', s=28, alpha=0.85,
        edgecolors='white', linewidths=0.4,
        label='Non-dominated front (analysis set)', zorder=3)
    ax0.scatter(
        [z_star[0]], [z_star[1]], c='#d62728', s=140, marker='*',
        zorder=5, edgecolors='k', linewidths=0.5,
        label=r'Ideal $z^*$ (component-wise min on front)')

    label_frac = 0.88
    for j in range(ref.shape[0]):
        col = cmap(gn(best_g[j]))
        e = z_star + ray_len * ref_unit[j]
        ax0.plot(
            [z_star[0], e[0]], [z_star[1], e[1]], color=col,
            lw=1.6, alpha=0.9, zorder=2)
        pos = z_star + label_frac * ray_len * ref_unit[j]
        txt = ax0.text(
            float(pos[0]), float(pos[1]), str(j),
            fontsize=7, ha='center', va='center', color='#1a1a1a', zorder=6)
        txt.set_path_effects([
            pe.Stroke(linewidth=2.5, foreground='white'),
            pe.Normal(),
        ])

    sm = ScalarMappable(cmap=cmap, norm=gn)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax0, shrink=0.82, pad=0.02)
    cbar.set_label(
        r'Best Tchebycheff $g$ per direction (higher = slower / worse)')

    ax0.set_xlabel(r'$f_1$')
    ax0.set_ylabel(r'$f_2$')
    ax0.set_title(
        r'Reference rays colored by $g_j^*=\min_i g^{\mathrm{te}}(f_i\mid\lambda_j)$')
    ax0.grid(True, alpha=0.35, linestyle='--')
    ax0.legend(loc='upper right', fontsize=9)
    ax0.set_aspect('equal', adjustable='box')

    jj = np.arange(ref.shape[0])
    ax1.barh(
        jj, best_g[jj], color=cmap(gn(best_g[jj])), edgecolor='0.35', linewidth=0.4)
    ax1.set_yticks(jj)
    ax1.set_yticklabels(
        [rf'$j={j}\ \lambda=({ref[j,0]:.2f},{ref[j,1]:.2f})$' for j in jj],
        fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$\min_i g^{\mathrm{te}}(f_i\mid\lambda_j)$ (lower is better)')
    ax1.set_title('Achievement per direction (same colormap as left)')
    topm = min(5, len(ordered))
    tops = ', '.join(str(int(ordered[k])) for k in range(topm))
    ax1.text(
        0.02, 0.02,
        f'Slowest direction indices (slowest first), top {topm}: {tops}',
        transform=ax1.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    fig.suptitle(
        'Das-Dennis direction progress on the Pareto front (`find_slow_directions`)',
        fontsize=12, fontweight='bold')

    out = Path(__file__).resolve().parent / 'slow_reference_directions_analysis.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
