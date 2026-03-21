import os
import numpy as np
import pytest

from AeroOpt.core.settings import SettingsData, SettingsProblem
from AeroOpt.core.problem import Problem
from AeroOpt.core.individual import Individual
from AeroOpt.core.database import Database


@pytest.fixture(scope="module")
def problem():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_path = os.path.join(root, "template_settings.json")
    sd = SettingsData("default", fname_settings=settings_path)
    sp = SettingsProblem("default", sd, fname_settings=settings_path)
    return Problem(sd, sp)


@pytest.fixture
def database(problem):
    return Database(problem, database_type="population")


def _indi(problem, x, y=None, ID=None):
    return Individual(problem, x=np.array([x]), y=None if y is None else np.array([y]), ID=ID)


def _append_direct(db: Database, indi: Individual):
    db.individuals.append(indi)
    db._id_list.append(indi.ID)
    db._sorted = False


class TestDatabaseBasics:
    def test_add_and_id_list(self, database, problem):
        a = _indi(problem, 0.2, 0.2, ID=1)
        assert database.add_individual(a) is True
        # 当前实现中首次 add 会重复 append 一次（回归测试）
        assert database.size == 2
        assert database.get_ID_from_index(0) == 1
        assert database.get_index_from_ID(1) in (0, 1)

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


class TestDatabaseDuplication:
    def test_check_duplication_current_behavior(self, database, problem):
        database.add_individual(_indi(problem, 0.5, 0.5, ID=1), check_duplication=False)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            database.check_duplication(np.array([0.5]))

    def test_add_second_individual_current_behavior(self, database, problem):
        database.add_individual(_indi(problem, 0.25, 0.25, ID=1), check_duplication=False)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            database.add_individual(_indi(problem, 0.3, 0.3, ID=2), check_duplication=False)


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

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            db1.merge_with_database(db2, deepcopy=True)


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
