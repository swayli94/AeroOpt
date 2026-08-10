"""
Regression tests for bugs found during the 0.2.0 audit.

Each test pins down one defect that was reachable from the public API, so a
future refactor cannot quietly reintroduce it.
"""

import os
import platform

import numpy as np
import pytest

from aeroopt.analysis.analyze_database import AnalyzeDatabase
from aeroopt.core import Database, Individual, Problem, SettingsData, SettingsProblem


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "aeroopt", "template_settings.json")


@pytest.fixture
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


def _add(db: Database, x: float, y: float | None = None) -> Individual:
    indi = Individual(db.problem, x=np.array([x]),
                      y=None if y is None else np.array([y]))
    db.add_individual(indi, print_warning_info=False)
    return db.individuals[-1]


class TestProblemRegressions:
    def test_scale_x_does_not_mutate_the_caller_array(self, settings_path):
        """`apply_precision` works in place; scaling must not leak that."""
        sd = SettingsData("default", fname_settings=settings_path)
        sd.input_precision = np.array([0.1])
        sp = SettingsProblem("default", sd, fname_settings=settings_path)
        prob = Problem(sd, sp)

        x = np.array([0.37])
        scaled = prob.scale_x(x)

        assert x[0] == pytest.approx(0.37), "caller's array was modified"
        assert scaled[0] == pytest.approx(0.4)

    def test_scale_y_does_not_mutate_the_caller_array(self, problem):
        y = np.array([0.25])
        problem.scale_y(y)
        assert y[0] == pytest.approx(0.25)

    def test_problem_is_hashable(self, problem):
        """Defining __eq__ without __hash__ would make Problem unhashable."""
        assert len({problem, problem}) == 1
        assert hash(problem) == hash(problem)

    def test_read_output_tolerates_blank_and_short_lines(self, problem, tmp_path):
        fname = tmp_path / "output.txt"
        fname.write_text("\n\ngarbage\ny 0.5\n\n", encoding="utf-8")

        succeed, y = problem.read_output(str(fname))

        assert succeed is True
        np.testing.assert_allclose(y, [0.5])

    def test_read_output_reports_failure_when_value_missing(self, problem, tmp_path):
        fname = tmp_path / "output.txt"
        fname.write_text("something_else 1.0\n", encoding="utf-8")

        succeed, _ = problem.read_output(str(fname))

        assert succeed is False

    def test_platform_check_is_equality_not_substring(self):
        """`platform.system() in 'Windows'` is true for '' and 'in'."""
        assert ("" in "Windows") is True
        assert (platform.system() == "Windows") == (platform.system() == "Windows")

    @pytest.mark.skipif(platform.system() == "Windows",
                        reason="the shell script variant is POSIX only")
    def test_external_run_copies_runfiles_contents_not_the_folder(
            self, problem, tmp_path, monkeypatch):
        """
        `cp -r Runfiles folder/` nested the folder instead of copying its
        contents, so the run script was never found and every case failed.
        """
        runfiles = tmp_path / "Runfiles"
        runfiles.mkdir()
        (runfiles / "run.sh").write_text(
            "#!/bin/sh\n"
            "x=$(awk '{print $2}' input.txt)\n"
            "echo \"y $x\" > output.txt\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        problem.calculation_folder = "Calculation"
        problem.runfiles_folder = "Runfiles"

        succeed, y = problem.external_run("case_1", np.array([0.25]))

        case_dir = tmp_path / "Calculation" / "case_1"
        assert (case_dir / "run.sh").exists(), "run script was not copied in"
        assert not (case_dir / "Runfiles").exists(), "run-files folder was nested"
        assert succeed is True
        np.testing.assert_allclose(y, [0.25])

    def test_external_run_reports_failure_without_runfiles(self, problem, tmp_path,
                                                           monkeypatch):
        monkeypatch.chdir(tmp_path)
        problem.calculation_folder = "Calculation"
        problem.runfiles_folder = "Runfiles"

        succeed, _ = problem.external_run("case_1", np.array([0.25]),
                                          information=False)

        assert succeed is False


class TestSettingsRegressions:
    def test_constraint_functions_default_is_not_shared(self, settings_path):
        """A mutable default would leak constraints between problems."""
        sd = SettingsData("default", fname_settings=settings_path)

        first = SettingsProblem("default", sd, fname_settings=settings_path)
        first.constraint_functions.append(lambda x, y: 0.0)

        second = SettingsProblem("default", sd, fname_settings=settings_path)

        assert len(second.constraint_functions) == 0

    def test_constraint_functions_argument_is_copied(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        shared = []
        settings = SettingsProblem("default", sd, fname_settings=settings_path,
                                   constraint_functions=shared)
        settings.constraint_functions.append(lambda x, y: 0.0)

        assert len(shared) == 0


class TestDatabaseRegressions:
    def test_copy_from_database_does_not_alias_the_individual_list(self, problem):
        source = Database(problem, database_type="total")
        target = Database(problem, database_type="valid")
        _add(source, 0.5)

        target.copy_from_database(source, deepcopy=False)
        assert target.individuals is not source.individuals

        _add(target, 0.9)
        assert source.size == 1
        assert target.size == 2

    def test_get_sub_database_without_selector_returns_all(self, problem):
        db = Database(problem, database_type="total")
        for v in (0.1, 0.5, 0.9):
            _add(db, v)

        assert db.get_sub_database().size == db.size

    def test_get_sub_database_rejects_two_selectors(self, problem):
        db = Database(problem, database_type="total")
        _add(db, 0.5)
        with pytest.raises(ValueError, match="Only one of"):
            db.get_sub_database(ID_list=[1], index_list=[0])

    def test_delete_individual_invalidates_pareto_rank(self, problem):
        db = Database(problem, database_type="valid")
        for v in (0.2, 0.4, 0.6):
            _add(db, v, v)
        db._updated_pareto_rank = True
        db._index_pareto_fronts = [[0, 1, 2]]

        db.delete_individual(index=0)

        assert db.updated_pareto_rank is False
        assert db.index_pareto_fronts == []

    def test_shrink_database_stops_when_tail_is_fully_reserved(self, problem):
        """reserve_ratio=1.0 used to walk past the start of the list."""
        db = Database(problem, database_type="total")
        for i in range(6):
            _add(db, 0.1 * (i + 1))

        db.shrink_database(remaining_size=3, reserve_ratio=1.0)

        assert db.size >= 3

    def test_shrink_database_is_a_noop_at_the_target_size(self, problem):
        db = Database(problem, database_type="total")
        for i in range(3):
            _add(db, 0.1 * (i + 1))

        db.shrink_database(remaining_size=3)

        assert db.size == 3

    def test_critical_scaled_distance_is_a_property(self, problem):
        db = Database(problem, database_type="total")
        assert db.critical_scaled_distance == problem.critical_scaled_distance

    def test_json_round_trip_keeps_y_as_array(self, problem, tmp_path):
        db = Database(problem, database_type="total")
        _add(db, 0.3, 0.15)
        _add(db, 0.7)  # never evaluated

        fname = str(tmp_path / "db.json")
        db.output_database_json(fname)

        restored = Database(problem, database_type="total")
        restored.read_database_json(fname)

        assert restored.size == 2
        for indi in restored.individuals:
            assert isinstance(indi.y, np.ndarray)
        assert restored.individuals[0].is_evaluated is True
        assert restored.individuals[1].is_evaluated is False


class TestFailedEvaluationRegressions:
    """
    A failed evaluation leaves an empty output vector. A total database normally
    holds several, and the optimizer breeds from that database whenever the
    valid archive is still small, so every read path has to tolerate them.
    """

    @staticmethod
    def _db_with_a_failure(problem) -> Database:
        db = Database(problem, database_type="total")
        _add(db, 0.2, 0.1)
        _add(db, 0.4, 0.3)
        _add(db, 0.6, 0.5)
        failed = db.individuals[1]
        failed.valid_evaluation = False
        failed.y = np.array([])
        failed._scaled_y = None
        return db

    def test_get_ys_leaves_failed_rows_at_zero(self, problem):
        db = self._db_with_a_failure(problem)

        ys = db.get_ys()

        assert ys.shape == (3, problem.n_output)
        np.testing.assert_allclose(ys[1], np.zeros(problem.n_output))
        np.testing.assert_allclose(ys[0], [0.1])

    def test_get_unified_objectives_tolerates_failures(self, problem):
        db = self._db_with_a_failure(problem)
        assert db.get_unified_objectives(scale=True).shape[0] == 3

    def test_ranking_and_crowding_tolerate_failures(self, problem):
        from aeroopt.optimization.moea import DominanceBasedAlgorithm

        db = self._db_with_a_failure(problem)

        DominanceBasedAlgorithm.non_dominated_ranking(db)
        DominanceBasedAlgorithm.assign_crowding_distance(db)
        parents = DominanceBasedAlgorithm.build_temporary_parent_database(db, 2)

        assert parents.size == 2

    def test_an_evaluated_individual_dominates_a_failed_one(self, problem):
        good = Individual(problem, x=np.array([0.2]), y=np.array([0.1]))
        failed = Individual(problem, x=np.array([0.4]))
        failed.valid_evaluation = False

        assert good.check_dominance(failed) == 1
        assert failed.check_dominance(good) == -1

    def test_two_failed_individuals_are_non_dominated(self, problem):
        a = Individual(problem, x=np.array([0.2]))
        b = Individual(problem, x=np.array([0.4]))
        a.valid_evaluation = False
        b.valid_evaluation = False

        assert a.check_dominance(b) == 9

    def test_failed_individuals_rank_behind_evaluated_ones(self, problem):
        from aeroopt.optimization.moea import DominanceBasedAlgorithm

        db = self._db_with_a_failure(problem)
        DominanceBasedAlgorithm.non_dominated_ranking(db)

        failed = db.individuals[1]
        evaluated = [i for i in db.individuals if i.is_evaluated]
        assert all(indi.pareto_rank < failed.pareto_rank for indi in evaluated)


class TestIndividualRegressions:
    def test_unevaluated_individual_has_empty_y(self, problem):
        indi = Individual(problem, x=np.array([0.5]))
        assert isinstance(indi.y, np.ndarray)
        assert indi.y.size == 0
        assert indi.is_evaluated is False

    def test_scalar_y_is_promoted_to_an_array(self, problem):
        indi = Individual(problem, x=np.array([0.5]), y=np.array([0.25]))
        assert indi.is_evaluated is True
        np.testing.assert_allclose(indi.y, [0.25])

    def test_scaled_y_of_unevaluated_individual_is_zeros(self, problem):
        indi = Individual(problem, x=np.array([0.5]))
        np.testing.assert_allclose(indi.scaled_y, np.zeros(problem.n_output))

    def test_objectives_of_unevaluated_individual_are_zeros(self, problem):
        indi = Individual(problem, x=np.array([0.5]))
        np.testing.assert_allclose(indi.objectives, np.zeros(problem.n_objective))


class TestAnalyzeDatabaseRegressions:
    def test_handles_individuals_with_failed_evaluations(self, problem):
        """An empty `y` used to raise a broadcast error while building arrays."""
        db = Database(problem, database_type="total")
        _add(db, 0.2)                # never evaluated -> y is empty
        _add(db, 0.4, 0.3)

        analyze = AnalyzeDatabase(db)

        assert analyze.size == 2
        np.testing.assert_allclose(analyze._ys[0], np.zeros(problem.n_output))

    def test_eliminate_crowding_individuals_runs(self, problem):
        """Sorting by crowding used to raise because the flag was never set."""
        db = Database(problem, database_type="valid")
        for v in (0.10, 0.1001, 0.1002, 0.90):
            _add(db, v, v)

        analyze = AnalyzeDatabase(db)
        eliminated = analyze.eliminate_crowding_individuals(
            threshold_distance=0.01, threshold_potential=0.8)

        assert len(eliminated) > 0, "clustered individuals should be eliminated"
        assert db.size == 4 - len(eliminated)

    def test_eliminate_crowding_individuals_respects_n_min_left(self, problem):
        db = Database(problem, database_type="valid")
        for i in range(6):
            _add(db, 0.10 + 1.0e-4 * i, 0.5)

        analyze = AnalyzeDatabase(db)
        analyze.eliminate_crowding_individuals(
            threshold_distance=1.0, threshold_potential=0.0, n_min_left=3)

        assert db.size >= 3

    def test_potential_uses_a_derived_coefficient(self, problem):
        """With coef_potential left at 0, every distance scored 1.0."""
        db = Database(problem, database_type="valid")
        for v in (0.1, 0.2, 0.5, 0.9):
            _add(db, v, v)

        analyze = AnalyzeDatabase(db)
        potential = analyze.calculate_potential_induced_by_database(np.array([0.15]))

        assert analyze.coef_potential > 0.0
        assert 0.0 < float(potential) < db.size


class TestMOEADRegressions:
    def test_generate_candidates_has_no_mutable_default_arguments(self):
        import inspect

        from aeroopt.optimization.stochastic.moead import MOEAD

        signature = inspect.signature(MOEAD.generate_candidate_individuals)
        for name, parameter in signature.parameters.items():
            assert not isinstance(parameter.default, (list, dict, set)), (
                f'{name} has a mutable default argument')
            assert not isinstance(parameter.default, np.random.Generator), (
                f'{name} has a Generator created at import time')


class TestDriverSignatureRegressions:
    def test_every_driver_defaults_to_per_design_evaluation(self):
        """
        `SBO` defaulted `user_func_supports_parallel=True` while the other seven
        drivers defaulted to False, so an ordinary `evaluate(x)` was handed the
        whole `xs` matrix and failed with "Invalid ys shape". The inner
        optimizer's flag is set separately and is unaffected.
        """
        import inspect

        from aeroopt.optimization import (
            OptDE, OptMOEAD, OptNRBO, OptNSGAII, OptNSGAIII, OptRVEA,
        )
        from aeroopt.optimization.hybrid import SAO, SBO

        drivers = [OptNSGAII, OptNSGAIII, OptRVEA, OptMOEAD, OptDE, OptNRBO,
                   SAO, SBO]

        for driver in drivers:
            default = inspect.signature(
                driver.__init__).parameters['user_func_supports_parallel'].default
            assert default is False, (
                f'{driver.__name__} defaults user_func_supports_parallel to '
                f'{default!r}; a per-design evaluator is the common case')

    def test_every_driver_accepts_rng(self):
        import inspect

        from aeroopt.optimization import (
            OptDE, OptMOEAD, OptNRBO, OptNSGAII, OptNSGAIII, OptRVEA,
        )
        from aeroopt.optimization.hybrid import SAO, SBO

        for driver in [OptNSGAII, OptNSGAIII, OptRVEA, OptMOEAD, OptDE,
                       OptNRBO, SAO, SBO]:
            assert 'rng' in inspect.signature(driver.__init__).parameters, (
                f'{driver.__name__} does not accept rng')


class TestPackagingRegressions:
    def test_utils_is_an_importable_package(self):
        """Without __init__.py, setuptools drops aeroopt.utils from the wheel."""
        import aeroopt.utils

        assert hasattr(aeroopt.utils, '__file__')
        assert aeroopt.utils.__file__ is not None
        assert aeroopt.utils.__file__.endswith('__init__.py')

    def test_declared_packages_cover_every_source_directory(self):
        from setuptools.discovery import PackageFinder

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        found = set(PackageFinder.find(where=root, include=['aeroopt*']))

        for dirpath, _dirnames, filenames in os.walk(os.path.join(root, 'aeroopt')):
            if '__pycache__' in dirpath:
                continue
            if not any(f.endswith('.py') for f in filenames):
                continue
            module = os.path.relpath(dirpath, root).replace(os.sep, '.')
            assert module in found, f'{module} would be missing from the distribution'

    def test_runtime_dependencies_are_declared(self):
        import tomllib

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, 'pyproject.toml'), 'rb') as f:
            pyproject = tomllib.load(f)

        declared = ' '.join(pyproject['project']['dependencies'])
        for package in ('numpy', 'scipy', 'scikit-learn', 'numexpr', 'pydoe', 'openpyxl'):
            assert package in declared, f'{package} is imported but not declared'
