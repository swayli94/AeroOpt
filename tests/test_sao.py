import os

import numpy as np
import pytest

from AeroOpt.core import Individual, Problem, SettingsData, SettingsProblem
from AeroOpt.optimization import SettingsDE, SettingsOptimization
from AeroOpt.optimization.hybrid.sao import SAO
from AeroOpt.utils.surrogate import SurrogateModel

from tests.test_sbo import _InnerOptShell, _StubSurrogate


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "AeroOpt", "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def optimization_settings(settings_path, tmp_path):
    s = SettingsOptimization("default", fname_settings=settings_path)
    s.population_size = 4
    s.max_iterations = 1
    s.working_directory = str(tmp_path)
    return s


def test_sao_has_rng_and_de_offspring_generation(
    problem, optimization_settings, settings_path,
):
    alg = SettingsDE("default", fname_settings=settings_path)
    surrogate = _StubSurrogate(problem)
    surrogate.train(
        np.ones((2, problem.n_input)),
        np.ones((2, problem.n_output)),
    )
    inner = _InnerOptShell(problem)
    sao = SAO(
        problem=problem,
        optimization_settings=optimization_settings,
        algorithm_settings=alg,
        surrogate=surrogate,
        opt_on_surrogate=inner,
        ratio_from_surrogate=0.5,
        user_func=None,
        mp_evaluation=None,
        rng=np.random.default_rng(7),
    )
    assert sao.rng is not None
    xv = np.array([0.2])
    n_out = problem.n_output
    for i in range(4):
        indi = Individual(
            problem=problem,
            x=xv.copy(),
            y=np.full(n_out, 0.1 * (i + 1), dtype=float),
            ID=i + 1,
        )
        sao.db_valid.add_individual(
            indi,
            check_duplication=False,
            check_bounds=True,
            deepcopy=False,
            print_warning_info=False,
        )
    sao.db_valid._is_valid_database = True
    sao.iteration = 1
    sao._generate_candidate_individuals_from_evolutionary_operators()
    assert sao.db_candidate.size > 0


def test_sao_super_passes_mp_evaluation_not_as_parallel_flag(
    problem, optimization_settings, settings_path,
):
    class _MpSentinel:
        pass

    alg = SettingsDE("default", fname_settings=settings_path)
    surrogate = _StubSurrogate(problem)
    surrogate.train(
        np.ones((2, problem.n_input)),
        np.ones((2, problem.n_output)),
    )
    sentinel = _MpSentinel()
    sao = SAO(
        problem=problem,
        optimization_settings=optimization_settings,
        algorithm_settings=alg,
        surrogate=surrogate,
        opt_on_surrogate=_InnerOptShell(problem),
        mp_evaluation=sentinel,
        rng=np.random.default_rng(0),
    )
    assert sao.mp_evaluation is sentinel
    assert sao.user_func_supports_parallel is False
