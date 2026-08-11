"""
Regression tests for bugs found during the 0.2.0 audit.

Each test pins down one defect that was reachable from the public API, so a
future refactor cannot quietly reintroduce it.
"""

import os
import platform
import time

import numpy as np
import pytest

from aeroopt.analysis.analyze_database import AnalyzeDatabase
from aeroopt.core import (
    Database, Individual, MultiProcessEvaluation, Problem, SettingsData,
    SettingsProblem, StaleCaseFolderError,
)


#* Worker functions for the parallel-evaluation tests. They must be importable
#* by name, because a ProcessPoolExecutor pickles them to reach its workers.

def _worker_ok(x, **kwargs):
    return True, np.array([float(x[0]) * 2.0])


def _worker_raises_above_half(x, **kwargs):
    if float(x[0]) > 0.5:
        raise RuntimeError('solver bridge rejected the design')
    return True, np.array([float(x[0]) * 2.0])


def _worker_sleeps(x, **kwargs):
    time.sleep(0.3)
    return True, np.array([float(x[0]) * 2.0])


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


def _grid_problem(precision, low=None, upp=None, n_input=3, n_output=1):
    """A problem whose input variables live on a precision grid."""
    sd = SettingsData.from_values(
        'grid',
        name_input=[f'x{i + 1}' for i in range(n_input)],
        input_low=[0.0] * n_input if low is None else low,
        input_upp=[1.0] * n_input if upp is None else upp,
        input_precision=precision,
        name_output=[f'y{i + 1}' for i in range(n_output)],
        output_low=[-1.0e3] * n_output, output_upp=[1.0e3] * n_output,
        output_precision=[0.0] * n_output,
        critical_scaled_distance=1.0e-8,
    )
    sp = SettingsProblem.from_values('grid', sd, output_type=[-1] * n_output,
                                     constraint_strings=[])
    return Problem(sd, sp)


def _is_on_grid(xs, precision) -> bool:
    xs = np.atleast_2d(np.asarray(xs, dtype=float))
    precision = np.asarray(precision, dtype=float)
    mask = precision != 0
    if not np.any(mask):
        return True
    residual = xs[:, mask] / precision[mask]
    return bool(np.allclose(residual, np.round(residual), atol=1e-9))


class TestInputPrecisionRegressions:
    """
    `input_precision` was only applied by `scale_x`, i.e. on the initial
    sampling path. Every offspring operator returned a continuous vector, so an
    integer-like variable (a rib count, an even ply number) reached the external
    solver as a fraction and the evaluation failed before it started --- for a
    binomial-crossover operator, on roughly `cross_rate` of all candidates.
    """

    PRECISION = [1.0, 0.1, 0.0]

    def test_de_trial_vectors_stay_on_the_precision_grid(self):
        from aeroopt.optimization.stochastic.de import DiffEvolution

        problem = _grid_problem(self.PRECISION, low=[2.0, 0.0, 0.0],
                                upp=[12.0, 1.0, 1.0])
        rng = np.random.default_rng(3)

        db = Database(problem, database_type='total')
        for x in problem.latin_hypercube_sampling(8, seed=3):
            db.add_individual(Individual(problem, x=x, y=np.array([float(np.sum(x))])),
                              print_warning_info=False)

        db_candidate = Database(problem, database_type='population')
        DiffEvolution.generate_candidate_individuals(
            db, db_candidate, population_size=8, iteration=1, rng=rng)

        assert db_candidate.size > 0
        assert _is_on_grid(db_candidate.get_xs(), self.PRECISION)

    def test_sbx_and_mutation_stay_on_the_precision_grid(self):
        from aeroopt.optimization.utils import polynomial_mutation, sbx_crossover

        problem = _grid_problem(self.PRECISION, low=[2.0, 0.0, 0.0],
                                upp=[12.0, 1.0, 1.0])
        rng = np.random.default_rng(5)

        x1 = np.array([3.0, 0.2, 0.31])
        x2 = np.array([9.0, 0.7, 0.82])

        for _ in range(20):
            child1, child2 = sbx_crossover(x1, x2, problem, cross_rate=1.0, rng=rng)
            mutated = polynomial_mutation(child1, problem, mut_rate=1.0, rng=rng)

            assert _is_on_grid(child1, self.PRECISION)
            assert _is_on_grid(child2, self.PRECISION)
            assert _is_on_grid(mutated, self.PRECISION)

    def test_nrbo_candidates_stay_on_the_precision_grid(self):
        from aeroopt.optimization.stochastic.nrbo import NRBO

        problem = _grid_problem(self.PRECISION, low=[2.0, 0.0, 0.0],
                                upp=[12.0, 1.0, 1.0])
        rng = np.random.default_rng(11)

        db = Database(problem, database_type='total')
        for x in problem.latin_hypercube_sampling(8, seed=11):
            db.add_individual(Individual(problem, x=x, y=np.array([float(np.sum(x))])),
                              print_warning_info=False)

        db_candidate = Database(problem, database_type='population')
        NRBO.generate_candidate_individuals(
            db, db_candidate, population_size=8, iteration=1,
            max_iterations=5, rng=rng)

        assert db_candidate.size > 0
        assert _is_on_grid(db_candidate.get_xs(), self.PRECISION)

    def test_driver_snaps_candidates_injected_by_a_hook(self, tmp_path):
        """
        The operators snap their own output, so the driver's safety net is what
        covers candidates that arrive another way: a pre-processing hook, a
        user-defined injection, or a new algorithm whose author forgot.
        """
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        problem = _grid_problem(self.PRECISION, low=[2.0, 0.0, 0.0],
                                upp=[12.0, 1.0, 1.0])
        evaluated = []

        def user_func(x):
            evaluated.append(x.copy())
            return True, np.array([float(np.sum(x))])

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=4, max_iterations=0,
                working_directory=str(tmp_path), seed=1,
                info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=user_func, save_result_files=False, logging=False,
        )

        opt.db_candidate.add_individual(
            Individual(problem, x=np.array([4.7, 0.34, 0.5])),
            print_warning_info=False)

        opt.evaluate_db_candidate()

        np.testing.assert_allclose(evaluated[0], [5.0, 0.3, 0.5])
        np.testing.assert_allclose(opt.db_candidate.individuals[0].x, [5.0, 0.3, 0.5])

    def test_sampling_a_subset_of_variables_stays_on_the_grid(self):
        """
        `latin_hypercube_sampling` snaps to the grid through `scale_x` when it
        samples the whole vector, but the named-subset path scaled the values by
        hand and skipped it.
        """
        problem = _grid_problem(self.PRECISION, low=[2.0, 0.0, 0.0],
                                upp=[12.0, 1.0, 1.0])

        samples = problem.latin_hypercube_sampling(
            6, sample_variables=['x1', 'x2'], seed=2)

        assert _is_on_grid(samples, self.PRECISION[:2])

    def test_update_x_refreshes_the_scaled_input(self, problem):
        """`scaled_x` is cached, and duplication checks read it."""
        indi = Individual(problem, x=np.array([0.25]))
        indi.update_x(np.array([0.75]))

        np.testing.assert_allclose(indi.x, [0.75])
        np.testing.assert_allclose(indi.scaled_x, problem.scale_x(np.array([0.75])))


class TestCandidateFolderReuseRegressions:
    """
    An external evaluation runs in `Calculation/<candidate ID>`, and
    `db_candidate` is emptied and refilled every iteration, so its IDs restarted
    at 1 in each one. From iteration 1 on, every candidate found the previous
    generation's `input.txt` already in place, skipped the solver and read back
    the *previous* design's `output.txt` --- results silently attached to the
    wrong `x`, which no downstream check can detect.
    """

    def _run_two_iterations(self, tmp_path):
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        problem = _grid_problem([0.0, 0.0, 0.0])

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=5, max_iterations=2,
                working_directory=str(tmp_path), seed=4,
                info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=lambda x: (True, np.array([float(np.sum(x**2))])),
            save_result_files=False, logging=False,
        )

        folders_per_iteration = []
        original = Database.evaluate_individuals

        def _record(self, *args, **kwargs):
            folders_per_iteration.append([indi.ID for indi in self.individuals])
            return original(self, *args, **kwargs)

        Database.evaluate_individuals = _record
        try:
            opt.main()
        finally:
            Database.evaluate_individuals = original

        return opt, folders_per_iteration

    def test_working_folder_names_are_never_reused(self, tmp_path):
        _opt, folders_per_iteration = self._run_two_iterations(tmp_path)

        assert len(folders_per_iteration) == 3, 'expected one initial population and two iterations'

        used = [ID for folders in folders_per_iteration for ID in folders]
        assert len(used) == len(set(used)), (
            f'a working folder was reused across iterations: {folders_per_iteration}')

    def test_evaluated_folder_name_matches_the_stored_individual(self, tmp_path):
        """The ID a case was evaluated under is the ID it keeps in `db_total`."""
        opt, folders_per_iteration = self._run_two_iterations(tmp_path)

        evaluated_ids = {ID for folders in folders_per_iteration for ID in folders}
        for indi in opt.db_total.individuals:
            assert indi.ID in evaluated_ids, (
                f'individual {indi.ID} was renumbered away from its working folder')

    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='the shell script variant is POSIX only')
    def test_external_study_records_each_result_against_its_own_design(self, tmp_path):
        """
        The end-to-end shape of the bug: run two iterations against a real
        external solver and check every stored `y` against the `x` its own case
        folder was run with. Before the fix, iteration 1 re-read iteration 0's
        outputs and stored them against the new designs.
        """
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        runfiles = tmp_path / 'Runfiles'
        runfiles.mkdir()
        (runfiles / 'run.sh').write_text(
            "#!/bin/sh\n"
            "awk '{s += $2} END {printf \"y1 %.9f\\n\", s}' input.txt > output.txt\n",
            encoding='utf-8',
        )

        problem = _grid_problem([0.0, 0.0], n_input=2)
        problem.calculation_folder = str(tmp_path / 'Calculation')
        problem.runfiles_folder = str(runfiles)

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=4, max_iterations=2, seed=9,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=None, save_result_files=False, logging=False,
        )
        opt.main()

        assert opt.db_total.size >= 8

        for indi in opt.db_total.individuals:
            case_dir = tmp_path / 'Calculation' / str(indi.ID)
            assert case_dir.is_dir(), f'no working folder for ID {indi.ID}'

            succeed, x_of_case = problem.read_input(str(case_dir / 'input.txt'))
            assert succeed
            np.testing.assert_allclose(
                x_of_case, indi.x, atol=1e-9,
                err_msg=f'ID {indi.ID} was evaluated with a different design')

            # The solver sums the inputs, so the stored y pins x to y directly.
            np.testing.assert_allclose(indi.y, [float(np.sum(indi.x))], atol=1e-8)

    def test_ids_are_not_reused_after_a_duplicate_is_rejected(self, problem, tmp_path):
        """
        `db_total.get_largest_ID() + 1` is not a safe allocator on its own: a
        candidate rejected as a duplicate never reaches `db_total`, so the next
        iteration would hand its ID --- and its working folder --- to a
        different design.
        """
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=2, max_iterations=0,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=lambda x: (True, np.array([float(x[0])])),
            save_result_files=False, logging=False,
        )

        for x in (0.2, 0.4):
            opt.db_candidate.add_individual(Individual(problem, x=np.array([x])),
                                            print_warning_info=False)
        opt.evaluate_db_candidate()
        first_ids = [indi.ID for indi in opt.db_candidate.individuals]

        # Only the first survives the merge; the second duplicates it.
        opt.db_candidate.individuals[1].update_x(np.array([0.2]))
        opt.update_total_and_valid_with_candidate()
        assert opt.db_total.size == 1

        opt.db_candidate.empty_database()
        opt.db_candidate.add_individual(Individual(problem, x=np.array([0.8])),
                                        print_warning_info=False)
        opt.evaluate_db_candidate()

        assert opt.db_candidate.individuals[0].ID not in first_ids


class TestExternalRunStaleFolderRegressions:
    """
    `external_run` skipped the solver whenever the case folder already held an
    input file, so a folder left by an earlier study --- or by the same study
    before a re-parameterization --- handed back that study's output for a
    completely different `x`, with nothing downstream able to notice.
    """

    def _prepare_case(self, tmp_path, x_recorded, y_recorded):
        case_dir = tmp_path / 'Calculation' / 'case_1'
        case_dir.mkdir(parents=True)
        (case_dir / 'input.txt').write_text(f'  x  {x_recorded:.9f}\n', encoding='utf-8')
        (case_dir / 'output.txt').write_text(f'  y  {y_recorded:.9f}\n', encoding='utf-8')
        return case_dir

    def _use_folders(self, problem, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        problem.calculation_folder = 'Calculation'
        problem.runfiles_folder = 'Runfiles'

    def test_matching_input_file_is_reused_without_rerunning(self, problem, tmp_path,
                                                             monkeypatch):
        """The restart behaviour this shortcut exists for must keep working."""
        self._prepare_case(tmp_path, x_recorded=0.25, y_recorded=7.0)
        # The run-files folder is absent, so a re-run could not have succeeded.
        self._use_folders(problem, tmp_path, monkeypatch)

        succeed, y = problem.external_run('case_1', np.array([0.25]),
                                          information=False)

        assert succeed is True
        np.testing.assert_allclose(y, [7.0])

    def test_stale_input_file_raises(self, problem, tmp_path, monkeypatch):
        case_dir = self._prepare_case(tmp_path, x_recorded=0.25, y_recorded=7.0)
        self._use_folders(problem, tmp_path, monkeypatch)

        with pytest.raises(StaleCaseFolderError) as excinfo:
            problem.external_run('case_1', np.array([0.75]), information=False)

        message = str(excinfo.value)
        assert 'case_1' in message
        assert 'Calculation' in message, 'the message must say what to clear'
        assert (case_dir / 'output.txt').exists(), 'the old result was destroyed'

    def test_input_file_of_a_renamed_variable_raises(self, problem, tmp_path,
                                                     monkeypatch):
        """
        A re-parameterization keeps the folder but changes what is in it. An
        input file that does not hold every current variable is not evidence of
        a prepared case.
        """
        case_dir = tmp_path / 'Calculation' / 'case_1'
        case_dir.mkdir(parents=True)
        (case_dir / 'input.txt').write_text('  x_old  0.250000000\n', encoding='utf-8')
        (case_dir / 'output.txt').write_text('  y  7.0\n', encoding='utf-8')
        self._use_folders(problem, tmp_path, monkeypatch)

        with pytest.raises(StaleCaseFolderError):
            problem.external_run('case_1', np.array([0.25]), information=False)

    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='the shell script variant is POSIX only')
    def test_rerun_stale_cases_re_prepares_the_folder(self, problem, tmp_path,
                                                      monkeypatch):
        runfiles = tmp_path / 'Runfiles'
        runfiles.mkdir()
        (runfiles / 'run.sh').write_text(
            "#!/bin/sh\n"
            "x=$(awk '{print $2}' input.txt)\n"
            "echo \"y $x\" > output.txt\n",
            encoding='utf-8',
        )
        case_dir = self._prepare_case(tmp_path, x_recorded=0.25, y_recorded=7.0)
        self._use_folders(problem, tmp_path, monkeypatch)
        problem.rerun_stale_cases = True

        succeed, y = problem.external_run('case_1', np.array([0.75]),
                                          information=False)

        assert succeed is True
        np.testing.assert_allclose(y, [0.75], atol=1e-9)
        assert '0.75' in (case_dir / 'input.txt').read_text(encoding='utf-8')

    def test_stale_output_is_not_returned_when_the_rerun_fails(self, problem, tmp_path,
                                                               monkeypatch):
        """A folder with no run script cannot produce a new result; the old one
        must not stand in for it."""
        case_dir = self._prepare_case(tmp_path, x_recorded=0.25, y_recorded=7.0)
        self._use_folders(problem, tmp_path, monkeypatch)
        problem.rerun_stale_cases = True

        succeed, _ = problem.external_run('case_1', np.array([0.75]),
                                          information=False)

        assert succeed is False
        assert not (case_dir / 'output.txt').exists()

    def test_driver_warns_before_running_into_an_old_calculation_folder(
            self, problem, tmp_path, monkeypatch):
        """The hint must come before any solver time is spent."""
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        (tmp_path / 'Calculation' / '1').mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        problem.calculation_folder = 'Calculation'

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=2, max_iterations=0,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=None, save_result_files=False, logging=False,
        )

        messages = []
        opt.log = lambda text, **kwargs: messages.append(text)
        opt._warn_about_existing_case_folders()

        assert any('already holds 1 case' in text for text in messages), messages


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason='the shell script variant and POSIX signals')
class TestSolverTimeoutRegressions:
    """
    `subprocess.run(timeout=...)` signals the direct child only, so the timeout
    killed the run *script* and left the solver it had started orphaned but
    running --- still holding a licence and a core, which is what the timeout
    existed to prevent. The timed-out case then had its `output.txt` read back,
    so a solver that wrote a result and hung afterwards was recorded as a
    success, and on the next attempt the case was skipped as "already prepared".
    """

    def _problem(self, tmp_path, script: str):
        runfiles = tmp_path / 'Runfiles'
        runfiles.mkdir(exist_ok=True)
        (runfiles / 'run.sh').write_text(script, encoding='utf-8')

        sd = SettingsData.from_values(
            'd', name_input=['x'], input_low=[0.0], input_upp=[1.0],
            name_output=['y'], output_low=[-10.0], output_upp=[10.0])
        sp = SettingsProblem.from_values('d', sd, output_type=[-1],
                                         constraint_strings=[])
        problem = Problem(sd, sp)
        problem.calculation_folder = str(tmp_path / 'Calculation')
        problem.runfiles_folder = str(runfiles)
        problem.kill_grace_period = 1.0
        return problem

    @staticmethod
    def _is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def test_timeout_kills_the_solver_the_script_started(self, tmp_path):
        # The script starts the "solver" as a child and waits for it, the way a
        # run script wraps abaqus or mpirun.
        problem = self._problem(tmp_path, (
            "#!/bin/sh\n"
            "sh -c 'echo $$ > solver.pid; sleep 60'\n"
        ))

        pid_file = tmp_path / 'Calculation' / 'case_1' / 'solver.pid'
        solver_pid = None
        try:
            succeed, _ = problem.external_run('case_1', np.array([0.25]),
                                              information=False, timeout=2.0)

            assert succeed is False
            assert pid_file.exists(), 'the solver never started; test is not meaningful'

            solver_pid = int(pid_file.read_text().strip())

            deadline = time.time() + 10.0
            while self._is_alive(solver_pid) and time.time() < deadline:
                time.sleep(0.1)

            assert not self._is_alive(solver_pid), (
                f'the solver (pid {solver_pid}) outlived the timeout')

        finally:
            if solver_pid is not None and self._is_alive(solver_pid):
                os.kill(solver_pid, 9)

    def test_timeout_is_a_failure_even_when_an_output_file_exists(self, tmp_path):
        # Writes a result, then hangs: the result describes a run that never
        # finished, so it must not be recorded.
        problem = self._problem(tmp_path, (
            "#!/bin/sh\n"
            "echo 'y 3.75' > output.txt\n"
            "sleep 60\n"
        ))

        succeed, _ = problem.external_run('case_1', np.array([0.25]),
                                          information=False, timeout=2.0)

        assert succeed is False
        assert (tmp_path / 'Calculation' / 'case_1' / problem.timeout_marker_fname).exists()

    def test_a_timed_out_case_is_re_run_not_skipped(self, tmp_path):
        problem = self._problem(tmp_path, (
            "#!/bin/sh\n"
            "echo 'y 3.75' > output.txt\n"
            "sleep 60\n"
        ))

        succeed, _ = problem.external_run('case_1', np.array([0.25]),
                                          information=False, timeout=2.0)
        assert succeed is False

        # The solver is fixed (or the machine is less loaded) and the study is
        # restarted with the same design.
        (tmp_path / 'Runfiles' / 'run.sh').write_text(
            "#!/bin/sh\n"
            "echo 'y 1.5' > output.txt\n",
            encoding='utf-8')

        succeed, y = problem.external_run('case_1', np.array([0.25]),
                                          information=False, timeout=10.0)

        assert succeed is True
        np.testing.assert_allclose(y, [1.5])
        assert not (tmp_path / 'Calculation' / 'case_1' / problem.timeout_marker_fname).exists()

    def test_a_finished_case_is_still_skipped_on_restart(self, tmp_path):
        """The restart shortcut must survive the timeout handling."""
        problem = self._problem(tmp_path, (
            "#!/bin/sh\n"
            "echo run >> runs.log\n"
            "echo 'y 1.5' > output.txt\n"
        ))

        for _ in range(3):
            succeed, y = problem.external_run('case_1', np.array([0.25]),
                                              information=False, timeout=10.0)
            assert succeed is True
            np.testing.assert_allclose(y, [1.5])

        runs = (tmp_path / 'Calculation' / 'case_1' / 'runs.log').read_text()
        assert runs.count('run') == 1, 'a finished case was re-run'

    def test_a_run_within_the_timeout_is_unaffected(self, tmp_path):
        problem = self._problem(tmp_path, (
            "#!/bin/sh\n"
            "sleep 0.2\n"
            "awk '{printf \"y %.9f\\n\", $2 * 2}' input.txt > output.txt\n"
        ))

        succeed, y = problem.external_run('case_1', np.array([0.25]),
                                          information=False, timeout=20.0)

        assert succeed is True
        np.testing.assert_allclose(y, [0.5], atol=1e-9)

    def test_solver_timeout_bounds_a_serial_study(self, tmp_path):
        """
        Without a `MultiProcessEvaluation` there was no way to bound an external
        run at all: `Database.evaluate_individuals` calls `external_run` with no
        timeout, so one hung solver hung the whole study forever.
        """
        problem = self._problem(tmp_path, "#!/bin/sh\nsleep 60\n")
        problem.solver_timeout = 2.0

        db = Database(problem, database_type='total')
        db.add_individual(Individual(problem, x=np.array([0.25]), ID=1),
                          print_warning_info=False)

        t0 = time.perf_counter()
        db.evaluate_individuals(mp_evaluation=None, user_func=None)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, 'the serial study waited for the hung solver'
        assert db.individuals[0].valid_evaluation is False
        assert db.individuals[0].y.size == 0

    def test_a_missing_run_script_is_a_failure_not_a_timeout(self, tmp_path):
        problem = self._problem(tmp_path, "#!/bin/sh\nexit 1\n")
        (tmp_path / 'Runfiles' / 'run.sh').unlink()

        succeed, _ = problem.external_run('case_1', np.array([0.25]),
                                          information=False, timeout=5.0)

        assert succeed is False
        assert not (tmp_path / 'Calculation' / 'case_1' / problem.timeout_marker_fname).exists()


class TestParallelEvaluationRegressions:
    """
    `MultiProcessEvaluation` used its per-evaluation `timeout` as the timeout of
    `as_completed`, i.e. of the whole batch. With more designs than processes
    the batch legitimately takes several times one evaluation, so the setting
    that protects against a hung solver reliably killed the study instead ---
    after the pool had finished the work, since leaving the `with` block waits.
    A worker that raised took the study down the same way.
    """

    def test_a_per_evaluation_timeout_does_not_cap_the_batch(self):
        xs = np.linspace(0.0, 1.0, 4).reshape(-1, 1)

        # Four 0.3 s designs through one process take ~1.2 s, well past the
        # 0.5 s any single one of them is allowed.
        mp = MultiProcessEvaluation(1, 1, func=_worker_sleeps, n_process=1,
                                    information=False, timeout=0.5)

        list_succeed, ys = mp.evaluate(xs)

        assert all(list_succeed)
        np.testing.assert_allclose(ys[:, 0], xs[:, 0] * 2.0)

    def test_a_raising_design_is_recorded_as_a_failure(self):
        xs = np.linspace(0.0, 1.0, 4).reshape(-1, 1)

        mp = MultiProcessEvaluation(1, 1, func=_worker_raises_above_half,
                                    n_process=2, information=False)

        list_succeed, ys = mp.evaluate(xs)

        assert list_succeed == [x[0] <= 0.5 for x in xs], list_succeed
        np.testing.assert_allclose(ys[0, 0], 0.0)

    def test_a_raising_design_is_recorded_as_a_failure_in_serial_too(self):
        xs = np.linspace(0.0, 1.0, 4).reshape(-1, 1)

        mp = MultiProcessEvaluation(1, 1, func=_worker_raises_above_half,
                                    n_process=None, information=False)

        list_succeed, _ = mp.evaluate(xs)

        assert list_succeed == [x[0] <= 0.5 for x in xs], list_succeed

    def test_batch_timeout_reports_failures_instead_of_raising(self):
        xs = np.linspace(0.0, 1.0, 3).reshape(-1, 1)

        mp = MultiProcessEvaluation(1, 1, func=_worker_sleeps, n_process=1,
                                    information=False, batch_timeout=0.05)

        list_succeed, ys = mp.evaluate(xs)

        assert len(list_succeed) == 3
        assert not all(list_succeed), 'the batch timeout did not fire'
        assert ys.shape == (3, 1)

    def test_a_healthy_batch_is_unaffected(self):
        xs = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

        mp = MultiProcessEvaluation(1, 1, func=_worker_ok, n_process=2,
                                    information=False)

        list_succeed, ys = mp.evaluate(xs)

        assert all(list_succeed)
        np.testing.assert_allclose(ys[:, 0], xs[:, 0] * 2.0)

    def test_a_stale_case_folder_still_stops_the_study(self, tmp_path, monkeypatch):
        """
        The failure handling must not swallow the one error that means the
        results are being read from another study.
        """
        sd = SettingsData.from_values(
            'd', name_input=['x'], input_low=[0.0], input_upp=[1.0],
            name_output=['y'], output_low=[-10.0], output_upp=[10.0])
        sp = SettingsProblem.from_values('d', sd, output_type=[-1],
                                         constraint_strings=[])
        problem = Problem(sd, sp)

        case_dir = tmp_path / 'Calculation' / 'case_1'
        case_dir.mkdir(parents=True)
        (case_dir / 'input.txt').write_text('  x  0.250000000\n', encoding='utf-8')
        monkeypatch.chdir(tmp_path)

        mp = MultiProcessEvaluation(1, 1, func=None, n_process=None,
                                    information=False)

        with pytest.raises(StaleCaseFolderError):
            mp.evaluate(np.array([[0.75]]), list_name=['case_1'], prob=problem)


class TestEvaluationBudgetRegressions:
    """
    An evaluation is the expensive thing in this framework, and two of them were
    being spent on designs that could not add anything: candidates duplicating
    one already in `db_total` were evaluated and only *then* rejected by the
    merge, and a resumed study re-sampled a whole initial population --- with a
    fixed seed, the identical one it had already evaluated.
    """

    def _study(self, tmp_path, counter, resume=False, **settings):
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        problem = _grid_problem([1.0, 0.1, 0.0], low=[2.0, 0.5, 0.0],
                                upp=[12.0, 2.0, 1.0])

        def user_func(x):
            counter.append(x.copy())
            return True, np.array([float(np.sum(x**2))])

        fields = dict(population_size=16, max_iterations=8, seed=1,
                      working_directory=str(tmp_path), info_level_on_screen=0,
                      resume=resume)
        fields.update(settings)

        return OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values('o', **fields),
            algorithm_settings=SettingsDE.from_values('a', cross_rate=0.9),
            user_func=user_func, save_result_files=True, logging=False,
        )

    def test_no_evaluation_is_spent_on_an_already_evaluated_design(self, tmp_path):
        evaluated = []
        opt = self._study(tmp_path, evaluated)
        opt.main()

        assert len(evaluated) == opt.db_total.size, (
            f'{len(evaluated) - opt.db_total.size} evaluations produced nothing')

    def test_resuming_does_not_re_sample_an_initial_population(self, tmp_path):
        import shutil

        first = []
        opt = self._study(tmp_path, first)
        opt.main()

        summary = tmp_path / 'Summary'
        shutil.copy(summary / 'db-total.json', summary / 'db-resume.json')

        second = []
        opt2 = self._study(tmp_path, second, resume=True)
        opt2.main()

        assert len(second) == opt2.db_total.size - opt.db_total.size, (
            'the resumed run paid for designs it already had')
        assert opt2.db_total.size > opt.db_total.size, 'the resumed run made no progress'

    def test_force_initial_population_size_still_samples_on_resume(self, tmp_path):
        import shutil

        first = []
        opt = self._study(tmp_path, first, max_iterations=0)
        opt.main()

        summary = tmp_path / 'Summary'
        shutil.copy(summary / 'db-total.json', summary / 'db-resume.json')

        second = []
        opt2 = self._study(tmp_path, second, max_iterations=0, resume=True,
                           force_initial_population_size=8, seed=99)
        opt2.main()

        assert len(second) > 0, 'an explicit initial population size was ignored'

    def test_a_generation_is_not_silently_short(self, tmp_path):
        """
        Operators dropped a slot whenever its offspring duplicated one already
        drawn, so on a coarse grid the generation --- and with it the search ---
        quietly shrank.
        """
        opt = self._study(tmp_path, [], max_iterations=0)
        opt.main()

        sizes = []
        for iteration in range(1, 6):
            opt.iteration = iteration
            opt.generate_candidate_individuals()
            sizes.append(opt.db_candidate.size)
            opt.evaluate_db_candidate()
            opt.update_total_and_valid_with_candidate()

        assert min(sizes) >= opt.population_size, sizes


class TestPostProcessRegressions:
    def test_pruning_in_post_process_is_reflected_in_the_same_iteration(self, tmp_path):
        """
        `db_valid` is derived from `db_total`, but it used to be derived before
        the post-processing hook that the documentation offers for pruning the
        archive --- so a pruned design was still selected as elite and written
        to that iteration's summary.
        """
        from aeroopt.optimization import (
            OptDE, PostProcess, SettingsDE, SettingsOptimization,
        )

        problem = _grid_problem([0.0, 0.0], n_input=2)

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=6, max_iterations=1, seed=3,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=lambda x: (True, np.array([float(np.sum(x**2))])),
            save_result_files=False, logging=False,
        )

        class PruneBest(PostProcess):
            """Drops whichever design is currently the best."""

            def apply(self) -> None:
                best = min(self.opt.db_total.individuals,
                           key=lambda indi: float(indi.y[0]))
                self.pruned_ID = best.ID
                self.opt.db_total.delete_individual(ID=best.ID)

        opt.post_process = PruneBest(opt)
        opt.main()

        pruned_ID = opt.post_process.pruned_ID
        assert pruned_ID not in [indi.ID for indi in opt.db_valid.individuals]
        assert pruned_ID not in [indi.ID for indi in opt.db_elite.individuals], (
            'a pruned design was still selected as elite')


class TestIndividualCopyRegressions:
    def test_copying_an_individual_shares_the_problem(self, problem):
        """
        `copy.deepcopy` cloned the whole `Problem` behind every individual ---
        its settings arrays and its constraint callables, once per individual,
        on every rebuild of `db_valid`. It also froze each copy against the
        problem as it was, and failed outright when a constraint callable held
        something uncopyable.
        """
        import copy

        indi = Individual(problem, x=np.array([0.5]), y=np.array([0.25]))
        clone = copy.deepcopy(indi)

        assert clone.problem is problem
        assert clone.x is not indi.x
        np.testing.assert_allclose(clone.x, indi.x)
        np.testing.assert_allclose(clone.y, indi.y)

    def test_a_whole_study_keeps_one_problem_object(self, tmp_path):
        from aeroopt.optimization import OptDE, SettingsDE, SettingsOptimization

        problem = _grid_problem([0.0, 0.0], n_input=2)

        opt = OptDE(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=6, max_iterations=2, seed=2,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsDE.from_values('a'),
            user_func=lambda x: (True, np.array([float(np.sum(x**2))])),
            save_result_files=False, logging=False,
        )
        opt.main()

        problems = {id(indi.problem) for indi in opt.db_total.individuals}
        problems |= {id(indi.problem) for indi in opt.db_valid.individuals}

        assert problems == {id(problem)}, (
            f'{len(problems)} problem objects for one study')

    def test_an_uncopyable_constraint_does_not_break_the_archive(self, settings_path):
        """A constraint callable may hold a model, a handle, a lock."""
        import copy
        import threading

        sd = SettingsData("default", fname_settings=settings_path)
        sp = SettingsProblem("default", sd, fname_settings=settings_path)
        prob = Problem(sd, sp)

        class _LockingConstraint:
            def __init__(self):
                self.lock = threading.Lock()

            def __call__(self, x, y):
                return 0.0

        sp.constraint_functions = [_LockingConstraint()]

        db = Database(prob, database_type='total')
        db.add_individual(Individual(prob, x=np.array([0.5]), y=np.array([0.25])),
                          print_warning_info=False)

        other = Database(prob, database_type='valid')
        other.copy_from_database(db, deepcopy=True)

        assert other.size == 1
        assert copy.deepcopy(db.individuals[0]).problem is prob


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
    def test_pending_replacements_point_at_the_offspring(self, tmp_path):
        """
        MOEA/D queues `(subproblem, offspring)` at generation time and applies
        the neighbour replacement after evaluation. It used to queue the
        offspring's *ID*, which is not stable: the driver renumbers
        `db_candidate` before evaluation (and the merge into `db_total` used to
        renumber it again), so the queued ID resolved to an individual of the
        initial population and the subproblem slots were rebound to the wrong
        designs.
        """
        from aeroopt.optimization import (
            OptMOEAD, SettingsMOEAD, SettingsOptimization,
        )

        problem = _grid_problem([0.0, 0.0], n_input=2, n_output=2)

        opt = OptMOEAD(
            problem=problem,
            optimization_settings=SettingsOptimization.from_values(
                'o', population_size=4, max_iterations=1, seed=6,
                working_directory=str(tmp_path), info_level_on_screen=0),
            algorithm_settings=SettingsMOEAD.from_values('a', n_partitions=3),
            user_func=lambda x: (True, np.array([float(np.sum(x**2)),
                                                 float(np.sum((1.0 - x)**2))])),
            save_result_files=False, logging=False,
        )

        opt.initialize_population()
        opt.iteration = 1
        opt.generate_candidate_individuals()

        xs_generated = opt.db_candidate.get_xs()

        opt.evaluate_db_candidate()
        opt.update_total_and_valid_with_candidate()

        assert len(opt._pending) == opt.db_candidate.size > 0

        for i, (_subproblem, offspring) in enumerate(opt._pending):
            # `getattr`: before the fix the queue held a bare ID.
            offspring_id = getattr(offspring, 'ID', offspring)
            index = opt.db_valid.get_index_from_ID(int(offspring_id))
            np.testing.assert_allclose(
                opt.db_valid.individuals[index].x, xs_generated[i],
                err_msg='the queued entry resolves to a different design')

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
