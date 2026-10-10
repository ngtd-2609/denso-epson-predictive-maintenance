"""Synthetic fixtures test software, not experimental fault-detection results."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline import audit, make_split, prepare


def fixture_data(root, n=4):
    root.mkdir()
    for label in ('A__D', 'AW__D'):
        for i in range(n):
            x = np.arange(24, dtype=float).reshape(8, 3) + i * 10
            if label == 'AW__D':
                x += 100
            np.savetxt(root / f'trimMA342_1200rpm_{label}_{i:02d}.csv', x, delimiter=',')


def test_audit_records_unknown_session_and_checks_duplicates(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    source = raw / 'trimMA342_1200rpm_A__D_00.csv'
    copy = raw / 'trimMA342_1200rpm_A__D_09.csv'
    copy.write_bytes(source.read_bytes())
    rows, report = audit(raw)
    assert len(rows) == 9
    assert all(r['session_id'] == 'unknown' for r in rows)
    assert rows[0]['sampling_rate_hz'] == 3000
    assert report['duplicate_signal_groups']
    assert all(r['n_samples'] == 8 for r in rows)


def test_bad_values_are_reported_and_cannot_enter_split(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    bad = raw / 'trimMA342_1200rpm_A__D_00.csv'
    bad.write_text('NaN,1,2\n3,4,5\n', encoding='utf-8')
    rows, report = audit(raw)
    assert report['invalid_records'] == 1
    with pytest.raises(ValueError, match='invalid'):
        make_split(rows, train_n=2, val_n=1, test_n=1, seed=42,
                   acknowledge_file_level=True)


def test_split_requires_acknowledgement_and_enough_records(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw, n=1)
    rows, _ = audit(raw)
    with pytest.raises(ValueError, match='provenance'):
        make_split(rows, 2, 1, 1, 42, acknowledge_file_level=False)
    with pytest.raises(ValueError, match='insufficient'):
        make_split(rows, 2, 1, 1, 42, acknowledge_file_level=True)


def test_duplicate_signal_cannot_cross_split(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    a = raw / 'trimMA342_1200rpm_A__D_00.csv'
    b = raw / 'trimMA342_1200rpm_A__D_01.csv'
    b.write_bytes(a.read_bytes())
    rows, _ = audit(raw)
    with pytest.raises(ValueError, match='duplicate'):
        make_split(rows, 2, 1, 1, 42, acknowledge_file_level=True)


def test_split_reproducible_disjoint(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    rows, _ = audit(raw)
    a = make_split(rows, 2, 1, 1, 42, acknowledge_file_level=True)
    b = make_split(list(reversed(rows)), 2, 1, 1, 42, acknowledge_file_level=True)
    assert {r['recording_id']: r['split'] for r in a} == {r['recording_id']: r['split'] for r in b}
    ids = [{r['recording_id'] for r in a if r['split'] == s} for s in ('train', 'validation', 'test')]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_scaler_uses_selected_train_and_windows_preserve_axes(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    rows, _ = audit(raw)
    split = make_split(rows, 2, 1, 1, 42, acknowledge_file_level=True)
    out = tmp_path / 'prepared'
    result = prepare(split, out, budgets=[1, 2], length=4, seed=7)
    assert set(result['budgets']['1']['fault_train_ids']).issubset(result['budgets']['2']['fault_train_ids'])
    for k in ('1', '2'):
        detail = result['budgets'][k]
        train_ids = detail['train_ids']
        selected = [r for r in split if r['recording_id'] in train_ids]
        real = np.concatenate([np.loadtxt(r['filepath'], delimiter=',', ndmin=2) for r in selected])
        np.testing.assert_allclose(detail['scaler']['mean'], real.mean(axis=0))
        assert all(r['split'] == 'train' for r in selected)
        data = np.load(out / f'k{k}' / 'train.npz')
        assert data['X'].shape[1:] == (3, 4)
        with (out / f'k{k}' / 'windows.csv').open(encoding='utf-8', newline='') as f:
            first = next(csv.DictReader(f))
        original = next(r for r in split if r['recording_id'] == first['recording_id'])
        x = np.loadtxt(original['filepath'], delimiter=',', ndmin=2)[:4]
        expected = ((x - detail['scaler']['mean']) / detail['scaler']['scale']).T
        np.testing.assert_allclose(data['X'][0], expected, rtol=1e-6, atol=1e-6)
    normal_ids = result['normal_only']['train_ids']
    assert all(r['class_id'] == 0 for r in split if r['recording_id'] in normal_ids)
    with pytest.raises(FileExistsError):
        prepare(split, out, budgets=[1], length=4, seed=7)


def test_prepare_rejects_modified_source(tmp_path):
    raw = tmp_path / 'raw'
    fixture_data(raw)
    rows, _ = audit(raw)
    split = make_split(rows, 2, 1, 1, 42, acknowledge_file_level=True)
    Path(split[0]['filepath']).write_text('1,2,3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        prepare(split, tmp_path / 'prepared', budgets=[1], length=4, seed=7)
