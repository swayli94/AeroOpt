# Tests for AeroOpt.core.utils
# Windows 路径在 check_folder 中已区分

import os
import numpy as np
import pytest

from AeroOpt.core.utils import compare_ndarray, init_log, log, check_folder


class TestCompareNdarray:
    def test_equal(self):
        assert compare_ndarray(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0

    def test_less(self):
        assert compare_ndarray(np.array([1, 2, 2]), np.array([1, 2, 3])) == -1
        assert compare_ndarray(np.array([1, 2]), np.array([1, 3])) == -1

    def test_greater(self):
        assert compare_ndarray(np.array([1, 2, 4]), np.array([1, 2, 3])) == 1
        assert compare_ndarray(np.array([2]), np.array([1])) == 1

    def test_different_length(self):
        with pytest.raises(ValueError, match="same shape"):
            compare_ndarray(np.array([1, 2]), np.array([1, 2, 3]))


class TestInitLog:
    def test_init_log_creates_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "logging.log")
            init_log(d, fname=fname)
            assert os.path.exists(fname)
            with open(fname, "r", encoding="utf-8") as f:
                text = f.read()
            assert "Time:" in text and "Result path:" in text


class TestLog:
    def test_log_no_file(self, capsys):
        log("hello", fname=None)
        out, _ = capsys.readouterr()
        assert "hello" in out

    def test_log_with_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "log.txt")
            log("world", fname=fname)
            assert os.path.exists(fname)
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
            assert "world" in content


class TestCheckFolder:
    def test_check_folder_calculation_not_exists(self):
        # ./Calculation/ 可能不存在
        exist = check_folder("_nonexistent_test_folder_12345")
        assert exist is False

    def test_check_folder_calculation_exists(self):
        # 若存在 Calculation 目录，用其下不存在的子目录名
        base = os.path.join(".", "Calculation")
        if not os.path.exists(base):
            exist = check_folder("_nonexistent_")
            assert exist is False
        else:
            # 随便一个不存在的名字
            exist = check_folder("_nonexistent_sub_12345_")
            assert exist is False
