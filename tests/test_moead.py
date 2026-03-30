import json
import os
import random

import numpy as np
import pytest

from AeroOpt.core import Database, Individual, Problem, SettingsData, SettingsProblem
from AeroOpt.optimization import (
    DominanceBasedAlgorithm,
    MOEAD,
    OptMOEAD,
    SettingsMOEAD,
    SettingsOptimization,
)
from AeroOpt.optimization.moea import DecompositionBasedAlgorithm


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
def problem_biobj(tmp_path):
    cfg = {
        "sd_bio": {
            "type": "SettingsData",
            "name": "bio",
            "name_input": ["x"],
            "name_output": ["f1", "f2"],
            "input_low": [0.0],
            "input_upp": [1.0],
            "input_precision": [1.0e-6],
            "output_low": [-1.0e6, -1.0e6],
            "output_upp": [1.0e6, 1.0e6],
            "output_precision": [0.0, 0.0],
            "critical_scaled_distance": 1.0e-6,
        },
        "sp_bio": {
            "type": "SettingsProblem",
            "name": "bio",
            "name_data_settings": "bio",
            "output_type": [-1, -1],
            "constraint_strings": [],
        },
    }
    path = os.path.join(tmp_path, "bio_settings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    sd = SettingsData("bio", fname_settings=path)
    sp = SettingsProblem("bio", sd, fname_settings=path)
    return Problem(sd, sp)


def _indi(problem: Problem, x: float, y, ID: int) -> Individual:
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    indi = Individual(problem, x=np.array([x]), y=y_arr, ID=ID)
    indi.eval_constraints()
    return indi


@pytest.fixture
def optimization_settings_moead(settings_path, tmp_path):
    '''
    population_size=4 matches Das–Dennis with 2 objectives and n_partitions=3 (4 weights).
    '''
    s = SettingsOptimization("default", fname_settings=settings_path)
    s.population_size = 4
    s.max_iterations = 2
    s.working_directory = str(tmp_path)
    return s


def test_settings_moead_reads_template(settings_path):
    s = SettingsMOEAD("default", fname_settings=settings_path)
    assert s.name == "default"
    assert s.cross_rate == pytest.approx(0.8)
    assert s.pow_sbx == pytest.approx(20.0)
    assert s.n_neighbors == 20
    assert s.prob_neighbor_mating == pytest.approx(0.9)
    assert s.decomposition == "auto"
    assert s.pbi_theta == pytest.approx(5.0)
    assert s.n_partitions is None


def test_opt_moead_raises_single_objective(problem, optimization_settings_moead, settings_path):
    algo = SettingsMOEAD("default", fname_settings=settings_path)
    with pytest.raises(ValueError, match="at least two objectives"):
        OptMOEAD(problem, optimization_settings_moead, algo)


def test_opt_moead_select_elite_from_valid(
        problem_biobj, optimization_settings_moead, settings_path):
    algo = SettingsMOEAD("default", fname_settings=settings_path)
    opt = OptMOEAD(problem_biobj, optimization_settings_moead, algo)
    opt.db_valid = Database(problem_biobj, database_type="valid")
    opt.db_valid.add_individual(
        Individual(problem_biobj, x=np.array([0.2]), y=np.array([0.3, 0.4]), ID=1),
        check_duplication=False,
        print_warning_info=False,
    )
    opt.db_valid.add_individual(
        Individual(problem_biobj, x=np.array([0.4]), y=np.array([0.5, 0.3]), ID=2),
        check_duplication=False,
        print_warning_info=False,
    )
    opt.db_valid._is_valid_database = True
    opt.select_elite_from_valid()
    assert opt.db_elite.size >= 1
    assert all(indi.pareto_rank == 1 for indi in opt.db_elite.individuals)


def test_generate_candidate_individuals_requires_valid_population(problem_biobj):
    db_valid = Database(problem_biobj, database_type="valid")
    db_candidate = Database(problem_biobj, database_type="population")
    ref = DecompositionBasedAlgorithm.das_dennis_reference_points(2, 3)
    neighbors = MOEAD.neighbor_indices(ref, 4)
    slot_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    ideal = np.array([0.0, 0.0])
    pending: list = []
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="No valid individuals"):
        MOEAD.generate_candidate_individuals(
            db_valid,
            db_candidate,
            ref_dirs=ref,
            neighbors=neighbors,
            slot_ids=slot_ids,
            prob_neighbor=0.9,
            decomposition_method="tchebicheff",
            pbi_theta=5.0,
            ideal=ideal,
            iteration=1,
            cross_rate=1.0,
            pow_sbx=20.0,
            mut_rate=1.0,
            pow_poly=20.0,
            rng=rng,
            pending_list=pending,
        )


def test_generate_candidate_individuals_builds_offspring(problem_biobj):
    random.seed(123)
    np.random.seed(123)
    ref = DecompositionBasedAlgorithm.das_dennis_reference_points(2, 3)
    assert ref.shape[0] == 4
    neighbors = MOEAD.neighbor_indices(ref, 20)
    slot_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    ideal = np.array([0.0, 0.0])
    pending: list = []

    db_valid = Database(problem_biobj, database_type="valid")
    for i, x in enumerate([0.05, 0.35, 0.65, 0.92], start=1):
        db_valid.add_individual(
            _indi(problem_biobj, x, [x * 0.5, (1.0 - x) * 0.3], i),
            check_duplication=False,
            print_warning_info=False,
        )
    db_valid._is_valid_database = True
    DominanceBasedAlgorithm.rank_pareto(db_valid)

    db_candidate = Database(problem_biobj, database_type="population")
    rng = np.random.default_rng(123)
    MOEAD.generate_candidate_individuals(
        db_valid,
        db_candidate,
        ref_dirs=ref,
        neighbors=neighbors,
        slot_ids=slot_ids,
        prob_neighbor=0.9,
        decomposition_method="tchebicheff",
        pbi_theta=5.0,
        ideal=ideal,
        iteration=2,
        cross_rate=1.0,
        pow_sbx=20.0,
        mut_rate=1.0,
        pow_poly=20.0,
        rng=rng,
        pending_list=pending,
    )

    assert db_candidate.size == 4
    assert len(pending) == 4
    for indi in db_candidate.individuals:
        assert indi.source == 'evolutionary_operator'
        assert indi.generation == 2
        assert problem_biobj.check_bounds_x(indi.x)


def test_neighbor_indices_includes_self_first_column():
    ref = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=float)
    nei = MOEAD.neighbor_indices(ref, n_neighbors=2)
    assert nei.shape == (3, 2)
    assert nei[0, 0] == 0
    assert nei[1, 0] == 1
    assert nei[2, 0] == 2


def test_decomposed_values_tchebicheff():
    F = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=float)
    w = np.array([[0.5, 0.5], [0.7, 0.3]], dtype=float)
    z = np.array([0.0, 0.0], dtype=float)
    g = DecompositionBasedAlgorithm.decomposed_values(F, w, z, "tchebicheff", 5.0)
    assert g.shape == (2, 2)
    assert np.all(g >= 0.0)
    np.testing.assert_allclose(
        np.diag(g),
        [
            max(0.5 * 1.0, 0.5 * 2.0),
            max(0.7 * 0.5, 0.3 * 1.5),
        ],
    )


def test_ensure_state_cycles_when_fewer_valid_than_subproblems(
        problem_biobj, tmp_path):
    '''
    4 个子问题、仅 2 个可行个体时，槽位应按归档下标循环复用，不报错。
    '''
    cfg = {
        "opt_c": {
            "type": "SettingsOptimization",
            "name": "defc",
            "resume": False,
            "population_size": 4,
            "max_iterations": 1,
            "fname_db_total": "db-total.json",
            "fname_db_elite": "db-elite.json",
            "fname_db_population": "db-population.json",
            "fname_db_resume": "db-resume.json",
            "fname_log": "optimization.log",
            "working_directory": str(tmp_path),
            "info_level_on_screen": 1,
            "critical_potential_x": 0.2,
        },
        "moead_c": {
            "type": "SettingsMOEAD",
            "name": "defc",
            "cross_rate": 0.8,
            "mut_rate": 0.8,
            "pow_sbx": 20.0,
            "pow_poly": 20.0,
            "n_partitions": 3,
            "n_neighbors": 2,
            "prob_neighbor_mating": 0.9,
            "decomposition": "tchebicheff",
            "pbi_theta": 5.0,
        },
    }
    path = os.path.join(tmp_path, "moead_cycle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    s = SettingsOptimization("defc", fname_settings=path)
    algo = SettingsMOEAD("defc", fname_settings=path)
    opt = OptMOEAD(problem_biobj, s, algo)
    opt.db_valid = Database(problem_biobj, database_type="valid")
    opt.db_valid.add_individual(
        _indi(problem_biobj, 0.1, [0.2, 0.3], 1),
        check_duplication=False,
        print_warning_info=False,
    )
    opt.db_valid.add_individual(
        _indi(problem_biobj, 0.9, [0.4, 0.5], 2),
        check_duplication=False,
        print_warning_info=False,
    )
    opt.db_valid._is_valid_database = True
    opt._ensure_state()
    assert opt._slot_ids is not None
    assert opt._slot_ids.shape == (4,)
    assert set(int(x) for x in opt._slot_ids) <= {1, 2}
    assert opt._ideal is not None
    assert opt._ideal.shape == (2,)


def test_opt_moead_init_population_mismatch_raises(problem_biobj, tmp_path):
    '''
    双目标下 n_partitions=2 仅产生 3 个 Das–Dennis 权重，与 population_size=4 不一致，应报错。
    '''
    cfg = {
        "opt_m": {
            "type": "SettingsOptimization",
            "name": "defmo",
            "resume": False,
            "population_size": 4,
            "max_iterations": 1,
            "fname_db_total": "db-total.json",
            "fname_db_elite": "db-elite.json",
            "fname_db_population": "db-population.json",
            "fname_db_resume": "db-resume.json",
            "fname_log": "optimization.log",
            "working_directory": str(tmp_path),
            "info_level_on_screen": 1,
            "critical_potential_x": 0.2,
        },
        "moead_m": {
            "type": "SettingsMOEAD",
            "name": "defmo",
            "cross_rate": 0.8,
            "mut_rate": 0.8,
            "pow_sbx": 20.0,
            "pow_poly": 20.0,
            "n_partitions": 2,
            "n_neighbors": 20,
            "prob_neighbor_mating": 0.9,
            "decomposition": "auto",
            "pbi_theta": 5.0,
        },
    }
    path = os.path.join(tmp_path, "moead_mismatch.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    s = SettingsOptimization("defmo", fname_settings=path)
    algo = SettingsMOEAD("defmo", fname_settings=path)
    with pytest.raises(ValueError, match="population_size must equal"):
        OptMOEAD(problem_biobj, s, algo)
