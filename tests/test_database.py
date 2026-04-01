import os

import numpy as np
import pytest

from aeroopt.core.settings import SettingsData, SettingsProblem
from aeroopt.core.problem import Problem
from aeroopt.core.individual import Individual, ID_UNASSIGNED
from aeroopt.core.database import Database, _json_dump_numpy_safe


@pytest.fixture(scope="module")
def settings_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "aeroopt", "template_settings.json")


@pytest.fixture(scope="module")
def problem(settings_path):
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def database(problem):
    return Database(problem, database_type="population")


def _indi(problem, x, y=None, ID: int = ID_UNASSIGNED):
    return Individual(problem, x=np.array([x]), y=None if y is None else np.array([y]), ID=ID)


def _append_direct(db: Database, indi: Individual):
    db.individuals.append(indi)
    db._id_list.append(indi.ID)
    db._sorted = False


class TestDatabaseBasics:
    def test_add_and_id_list(self, database, problem):
        a = _indi(problem, 0.2, 0.2, ID=1)
        assert database.add_individual(a)[0] is True
        assert database.size == 1
        assert database.get_ID_from_index(0) == 1
        assert database.get_index_from_ID(1) == 0

    def test_get_xs_ys(self, database, problem):
        _append_direct(database, _indi(problem, 0.3, 0.4, ID=10))
        _append_direct(database, _indi(problem, 0.7, 0.9, ID=11))
        xs = database.get_xs()
        ys = database.get_ys()
        assert xs.shape == (2, 1)
        assert ys.shape == (2, 1)
        np.testing.assert_allclose(xs[:, 0], [0.3, 0.7])
        np.testing.assert_allclose(ys[:, 0], [0.4, 0.9])

    def test_delete_by_id_and_index(self, database, problem):
        _append_direct(database, _indi(problem, 0.1, 0.1, ID=5))
        _append_direct(database, _indi(problem, 0.4, 0.4, ID=6))
        database.delete_individual(ID=5)
        assert database.size == 1
        database.delete_individual(index=0)
        assert database.size == 0

    def test_add_individual_assigns_reliable_ids_for_unassigned_default(self, database, problem):
        # Seed one explicit ID so default-ID assignment must continue from current max ID.
        assert database.add_individual(
            _indi(problem, 0.15, 0.15, ID=7), check_duplication=False
        )[0] is True

        assert database.add_individual(
            _indi(problem, 0.35, 0.35), check_duplication=False
        )[0] is True
        assert database.add_individual(
            _indi(problem, 0.55, 0.55), check_duplication=False
        )[0] is True

        assigned_ids = [indi.ID for indi in database.individuals]
        assert assigned_ids == [7, 8, 9]
        assert ID_UNASSIGNED not in assigned_ids


class TestDatabaseDuplication:
    def test_check_duplication_detects_same_x(self, database, problem):
        database.add_individual(_indi(problem, 0.5, 0.5, ID=1), check_duplication=False)
        is_dup, closest = database.check_duplication(np.array([0.5]))
        assert is_dup is True
        assert closest == 0

    def test_add_second_distinct_x_with_duplication_check_off(self, database, problem):
        assert database.add_individual(
            _indi(problem, 0.25, 0.25, ID=1), check_duplication=False
        )[0] is True
        assert database.add_individual(
            _indi(problem, 0.3, 0.3, ID=2), check_duplication=False
        )[0] is True
        assert database.size == 2

    def test_add_rejected_when_duplicate_and_check_on(self, database, problem):
        database.add_individual(_indi(problem, 0.4, 0.4, ID=1), check_duplication=False)
        assert database.add_individual(_indi(problem, 0.4, 0.4, ID=2))[0] is False
        assert database.size == 1


class TestDatabaseSubAndMerge:
    def test_get_sub_database(self, database, problem):
        _append_direct(database, _indi(problem, 0.2, 0.2, ID=2))
        _append_direct(database, _indi(problem, 0.6, 0.6, ID=6))
        sub_db = database.get_sub_database(ID_list=[6], deepcopy=True)
        assert sub_db.size == 1
        assert sub_db.get_ID_from_index(0) == 6
        assert sub_db.database_type == "sub-database"

    def test_intersection_and_merge(self, problem):
        db1 = Database(problem, database_type="population")
        db2 = Database(problem, database_type="population")
        _append_direct(db1, _indi(problem, 0.2, 0.2, ID=1))
        _append_direct(db1, _indi(problem, 0.8, 0.8, ID=2))
        _append_direct(db2, _indi(problem, 0.8, 0.8, ID=20))
        _append_direct(db2, _indi(problem, 0.9, 0.9, ID=21))

        inter = db1.get_intersection_with_database(db2, deepcopy=True)
        assert inter.size == 1
        np.testing.assert_allclose(inter.individuals[0].x, [0.8])

        db1.merge_with_database(db2, deepcopy=True)
        assert db1.size == 3
        xs = sorted(db1.get_xs()[:, 0].tolist())
        np.testing.assert_allclose(xs, [0.2, 0.8, 0.9], rtol=0, atol=1e-12)


class TestDatabaseJsonIO:
    def test_output_and_read_json(self, problem, tmp_path):
        db = Database(problem, database_type="total")
        _append_direct(db, _indi(problem, 0.12, 0.34, ID=3))
        _append_direct(db, _indi(problem, 0.56, 0.78, ID=4))
        f = tmp_path / "db.json"
        db.output_database_json(str(f))

        db2 = Database(problem, database_type="default")
        db2.read_database_json(str(f))
        assert db2.size == 2
        assert db2.database_type == "total"
        np.testing.assert_allclose(db2.get_xs()[:, 0], [0.12, 0.56], rtol=0, atol=1e-12)

    def test_output_json_handles_numpy_scalar_and_array_fields(self, problem, tmp_path):
        db = Database(problem, database_type="total")
        indi = _indi(problem, 0.12, 0.34, ID=3)
        indi.sum_violation = np.float64(0.0)
        indi.group = int(7)
        _append_direct(db, indi)
        f = tmp_path / "db_numpy.json"

        db.output_database_json(str(f))

        db_json = f.read_text(encoding="utf-8")
        assert '"group": 7' in db_json
        assert '"sum_violation": 0.0' in db_json

    def test_json_dump_numpy_safe_composes_user_default(self):
        payload = {
            "numpy_value": np.float64(1.25),
            "custom_value": complex(3.0, 4.0),
        }

        def user_default(value):
            if isinstance(value, complex):
                return {"real": value.real, "imag": value.imag}
            raise TypeError

        import io
        buffer = io.StringIO()
        _json_dump_numpy_safe(payload, buffer, default=user_default)
        text = buffer.getvalue()
        assert '"numpy_value": 1.25' in text
        assert '"real": 3.0' in text
        assert '"imag": 4.0' in text


class TestDatabaseValidation:
    def test_invalid_database_type_raises(self, problem):
        with pytest.raises(ValueError, match="Invalid database type"):
            Database(problem, database_type="no-such-type")

    def test_copy_from_database_requires_same_problem_instance(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        sp = SettingsProblem("default", sd, fname_settings=settings_path)
        p_a = Problem(sd, sp)
        p_b = Problem(sd, sp)
        assert p_a is not p_b

        db_a = Database(p_a, database_type="population")
        db_b = Database(p_b, database_type="population")
        _append_direct(db_b, _indi(p_b, 0.5, 0.5, ID=1))

        with pytest.raises(ValueError, match="same problem"):
            db_a.copy_from_database(db_b)

    def test_get_sub_database_both_id_and_index_raises(self, database, problem):
        _append_direct(database, _indi(problem, 0.3, 0.3, ID=1))
        with pytest.raises(ValueError, match="Only one of"):
            database.get_sub_database(ID_list=[1], index_list=[0])

    def test_merge_intersection_require_same_problem(self, settings_path):
        sd = SettingsData("default", fname_settings=settings_path)
        sp = SettingsProblem("default", sd, fname_settings=settings_path)
        p_a = Problem(sd, sp)
        p_b = Problem(sd, sp)
        db_a = Database(p_a, database_type="population")
        db_b = Database(p_b, database_type="population")

        with pytest.raises(ValueError, match="same problem"):
            db_a.merge_with_database(db_b)

        with pytest.raises(ValueError, match="same problem"):
            db_a.get_intersection_with_database(db_b)

    def test_merge_with_empty_other_is_noop(self, problem):
        db = Database(problem, database_type="population")
        db.add_individual(_indi(problem, 0.2, 0.2, ID=1), check_duplication=False)
        empty = Database(problem, database_type="population")
        db.merge_with_database(empty, deepcopy=True)
        assert db.size == 1


class TestDatabaseEvaluateIndividuals:
    def test_evaluate_individuals_with_mp_sets_user_func(self, problem):
        class _FakeMPE:
            func = None

            def evaluate(self, xs, list_name=None, **kwargs):
                n = xs.shape[0]
                ys = np.array([[float(xs[i, 0]) * 3.0] for i in range(n)])
                assert self.func is not None
                return [True] * n, ys

        db = Database(problem, database_type="total")
        db.add_individual(_indi(problem, 0.2, 0.0, ID=1), check_duplication=False)

        def user_func(x):
            return True, np.array([0.0])

        db.evaluate_individuals(mp_evaluation=_FakeMPE(), user_func=user_func)
        assert db.individuals[0].valid_evaluation is True
        np.testing.assert_allclose(db.individuals[0].y, [0.6])

    def test_evaluate_individuals_raises_when_id_is_negative(self, problem):
        db = Database(problem, database_type="total")
        indi = Individual(problem, x=np.array([0.1]), ID=ID_UNASSIGNED)
        _append_direct(db, indi)
        with pytest.raises(ValueError, match=f"Individual ID is negative: {ID_UNASSIGNED}"):
            db.evaluate_individuals(mp_evaluation=None, user_func=lambda x: (True, np.array([0.0])))

    def test_evaluate_individuals_parallel_user_func_matrix(self, problem):
        def user_func_batch(xs):
            assert xs.ndim == 2
            assert xs.shape[1] == problem.n_input
            n = xs.shape[0]
            ys = (xs[:, 0:1] * 2.0)
            return [True] * n, ys

        db = Database(problem, database_type="total")
        db.add_individual(_indi(problem, 0.1, None, ID=1), check_duplication=False)
        db.add_individual(_indi(problem, 0.4, None, ID=2), check_duplication=False)

        db.evaluate_individuals(
            mp_evaluation=None,
            user_func=user_func_batch,
            user_func_supports_parallel=True,
        )
        np.testing.assert_allclose(db.individuals[0].y, [0.2])
        np.testing.assert_allclose(db.individuals[1].y, [0.8])
        assert all(indi.valid_evaluation for indi in db.individuals)

    def test_evaluate_individuals_parallel_user_func_wrong_ys_shape_raises(self, problem):
        def user_func_bad(xs):
            n = xs.shape[0]
            return [True] * n, np.zeros((n, problem.n_output + 1))

        db = Database(problem, database_type="total")
        db.add_individual(_indi(problem, 0.2, None, ID=1), check_duplication=False)
        with pytest.raises(ValueError, match="Invalid ys shape"):
            db.evaluate_individuals(
                mp_evaluation=None,
                user_func=user_func_bad,
                user_func_supports_parallel=True,
            )

    def test_evaluate_individuals_parallel_user_func_partial_failure(self, problem):
        def user_func_mixed(xs):
            n = xs.shape[0]
            ys = np.ones((n, problem.n_output))
            return [True, False], ys

        db = Database(problem, database_type="total")
        db.add_individual(_indi(problem, 0.1, None, ID=1), check_duplication=False)
        db.add_individual(_indi(problem, 0.2, None, ID=2), check_duplication=False)

        db.evaluate_individuals(
            mp_evaluation=None,
            user_func=user_func_mixed,
            user_func_supports_parallel=True,
        )
        assert db.individuals[0].valid_evaluation is True
        np.testing.assert_allclose(db.individuals[0].y, [1.0])
        assert db.individuals[1].valid_evaluation is False
        # New behavior: failed batched evaluation keeps y as empty ndarray (not None).
        assert isinstance(db.individuals[1].y, np.ndarray)
        assert db.individuals[1].y.size == 0
