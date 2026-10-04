"""Portable handoff tests use synthetic arrays, not detector benchmarks."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff import array_signature, rebase_rows, verify_arrays


def test_rebase_preserves_frozen_metadata(tmp_path):
    rows = [{"recording_id": "record_1", "filepath": "D:/old/file.csv", "class_id": 2}]
    original = copy.deepcopy(rows)
    result = rebase_rows(rows, tmp_path)
    assert rows == original
    assert result[0]["class_id"] == 2
    assert result[0]["filepath"] == str((tmp_path / "record_1.csv").resolve())
    with pytest.raises(ValueError):
        rebase_rows([{"recording_id": "../escape"}], tmp_path)


def test_exact_array_check_rejects_changed_content(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    expected = {"train.npz": {"arrays": {"X": array_signature(x)}}}
    np.savez(tmp_path / "train.npz", X=x)
    verify_arrays(tmp_path, expected)
    x[0, 0, 0] += 1
    np.savez(tmp_path / "train.npz", X=x)
    with pytest.raises(ValueError, match="Array content differs"):
        verify_arrays(tmp_path, expected)
