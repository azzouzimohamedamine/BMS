"""File-name resolution of the raw loader (same file renamed by OS/browser is accepted; others never)."""
import tempfile
from pathlib import Path

from bms.data.loader import _norm, find_raw_file


def test_norm_ignores_case_copy_suffix_and_extension():
    assert _norm("11_5_2015_Low_Current_OCV_test_SP20-1 (1).xlsx") == _norm("11_5_2015_low_current_OCV_test_SP20-1.xlsx")
    assert _norm("a.xls") == _norm("A.xlsx")


def test_find_raw_file_variants_and_no_wrong_substitution():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); (d / "sub").mkdir()
        (d / "sub" / "11_5_2015_low_current_OCV_test_SP20-1 (1).xlsx").write_bytes(b"x")
        (d / "11_21_2015_low_current_OCV_test_SP20-1.xlsx").write_bytes(b"x")
        assert find_raw_file("11_5_2015_low_current_OCV_test_SP20-1.xlsx", d).name.endswith("(1).xlsx")
        assert find_raw_file("10_16_2015_Initial_capacity_SP20-1.xlsx", d) is None   # different dataset: never substituted
        assert find_raw_file("nonexistent.xlsx", d / "no_such_dir") is None
