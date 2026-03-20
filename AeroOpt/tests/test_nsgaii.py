import os
import random

import numpy as np
import pytest

from AeroOpt.core.database import Database
from AeroOpt.core.individual import Individual
from AeroOpt.core.problem import Problem
from AeroOpt.core.settings import (
    SettingsData,
    SettingsNSGAII,
    SettingsOptimization,
    SettingsProblem,
)
from AeroOpt.optimization.stochastic.nsgaii import NSGAII


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "template_settings.json")
    assert os.path.exists(path), f"template_settings.json not found at {path}"
    return path


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    p = Problem(sd, sp)
    # Individual.objectives 依赖 problem.output_type
    p.output_type = sp.output_type
    return p


@pytest.fixture
def opt(problem, settings_path, tmp_path):
    opt_settings = SettingsOptimization("default", fname_settings=settings_path)
    opt_settings.population_size = 4
    opt_settings.max_iterations = 3
    opt_settings.working_directory = str(tmp_path)

    nsgaii_settings = SettingsNSGAII("default", fname_settings=settings_path)
    # 绕开 OptBaseFramework.__init__ 对空数据库统计的依赖，仅测试 NSGA-II 算法逻辑。
    opt = object.__new__(NSGAII)
    opt.problem = problem
    opt.optimization_settings = opt_settings
    opt.nsgaii_settings = nsgaii_settings
    opt.db_total = Database(problem, database_type="total")
    opt.db_valid = Database(problem, database_type="valid")
    opt.db_elite = Database(problem, database_type="elite")
    opt.db_candidate = Database(problem, database_type="population")
    return opt


def _make_valid_individual(problem: Problem, x: float, y: float, ID: int) -> Individual:
    indi = Individual(problem, x=np.array([x]), y=np.array([y]), ID=ID)
    indi.eval_constraints()
    return indi


def _append_individuals_without_db_checks(db: Database, individuals):
    db.individuals.extend(individuals)
    db.update_id_list()


class TestNSGAIIOperators:
    def test_sbx_crossover_bounds(self, problem):
        random.seed(42)
        np.random.seed(42)
        x1 = np.array([0.2])
        x2 = np.array([0.8])
        c1, c2 = NSGAII.sbx_crossover(x1, x2, problem, cross_rate=1.0, pow_sbx=20.0)
        assert c1.shape == (1,)
        assert c2.shape == (1,)
        assert problem.check_bounds_x(c1)
        assert problem.check_bounds_x(c2)

    def test_polynomial_mutation_bounds(self, problem):
        random.seed(123)
        np.random.seed(123)
        x = np.array([0.5])
        out = NSGAII.polynomial_mutation(x, problem, mut_rate=1.0, pow_poly=20.0)
        assert out.shape == (1,)
        assert problem.check_bounds_x(out)


class TestNSGAIIRankingAndSelection:
    def test_fast_non_dominated_sort_single_objective(self, opt, problem):
        db = Database(problem, database_type="valid")
        # 单目标最小化：y 越小越优，预期 rank 依次增加
        _append_individuals_without_db_checks(
            db,
            [
                _make_valid_individual(problem, 0.2, 0.15, 1),
                _make_valid_individual(problem, 0.4, 0.35, 2),
                _make_valid_individual(problem, 0.6, 0.55, 3),
                _make_valid_individual(problem, 0.8, 0.75, 4),
            ],
        )

        fronts = opt.fast_non_dominated_sort(db)
        assert len(fronts) >= 4
        assert fronts[0] == [0]
        assert db.individuals[0].pareto_rank == 1
        assert db.individuals[3].pareto_rank == 4

    def test_assign_crowding_distance(self, opt, problem):
        db = Database(problem, database_type="valid")
        _append_individuals_without_db_checks(
            db,
            [_make_valid_individual(problem, v, v - 0.05, i) for i, v in enumerate([0.2, 0.4, 0.6, 0.8], start=1)],
        )

        fronts = [[0, 1, 2, 3]]
        opt._assign_crowding_distance(db, fronts)
        assert np.isinf(db.individuals[0].crowding_distance)
        assert np.isinf(db.individuals[3].crowding_distance)
        assert db.individuals[1].crowding_distance >= 0.0
        assert db.individuals[2].crowding_distance >= 0.0

    def test_binary_tournament_selection_size(self, opt, problem):
        random.seed(7)
        db = Database(problem, database_type="valid")
        individuals = []
        for i, v in enumerate([0.2, 0.4, 0.6, 0.8], start=1):
            indi = _make_valid_individual(problem, v, v - 0.05, i)
            indi.pareto_rank = i
            indi.crowding_distance = float(5 - i)
            individuals.append(indi)
        _append_individuals_without_db_checks(db, individuals)
        db.sort_database(sort_type=0)

        selected = opt.binary_tournament_selection(db, 4)
        assert len(selected) == 4
        assert all(isinstance(indi, Individual) for indi in selected)

    def test_select_valid_elite_from_total(self, opt, problem, monkeypatch):
        opt.db_total = Database(problem, database_type="total")
        _append_individuals_without_db_checks(
            opt.db_total,
            [_make_valid_individual(problem, v, v - 0.05, i) for i, v in enumerate([0.2, 0.4, 0.6, 0.8, 0.9], start=1)],
        )

        def _patched_add_individual(self, indi, **kwargs):
            self.individuals.append(indi)
            self._id_list.append(indi.ID)
            self._sorted = False
            return True

        monkeypatch.setattr(Database, "add_individual", _patched_add_individual)
        opt.select_valid_elite_from_total()
        assert opt.db_valid.size == opt.population_size
        assert opt.db_elite.size >= 1
        assert opt.db_elite.individuals[0].pareto_rank == 1
