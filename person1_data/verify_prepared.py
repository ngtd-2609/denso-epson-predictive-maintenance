"""Structural integration checks. Never evaluates a model or tunes on test."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from data_pipeline import digest, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--split', type=Path, required=True)
args = parser.parse_args()
report = json.loads((args.prepared / 'preparation_report.json').read_text(encoding='utf-8'))
rows = json.loads(args.split.read_text(encoding='utf-8'))
by_id = {r['recording_id']: r for r in rows}
profiles = {f'k{k}': v for k, v in report['budgets'].items()}
profiles['normal_only'] = report['normal_only']
checks = {}
previous_faults = set()
for k in sorted(report['budgets'], key=int):
    current = set(report['budgets'][k]['fault_train_ids'])
    assert previous_faults <= current
    previous_faults = current
for name, detail in profiles.items():
    expected_train = set(detail['train_ids'])
    assert set(detail['scaler']['fit_recording_ids']) == expected_train
    assert all(by_id[r]['split'] == 'train' for r in expected_train)
    hashes, collisions, counts, split_ids = {}, [], {}, {}
    for split in ('train', 'validation', 'test'):
        with np.load(args.prepared / name / f'{split}.npz', allow_pickle=False) as data:
            x, labels, ids, starts = data['X'], data['y_class'], data['recording_id'], data['start_sample']
            assert x.dtype == np.float32 and x.shape[1:] == (3, report['length'])
            assert np.isfinite(x).all()
            assert len(x) == len(labels) == len(ids) == len(starts)
            expected = expected_train if split == 'train' else {r['recording_id'] for r in rows if r['split'] == split}
            assert set(ids) == expected
            assert np.all(data['y_binary'] == (labels != 0))
            for label, rid, start in zip(labels, ids, starts):
                assert label == by_id[rid]['class_id']
                assert start % report['length'] == 0
                assert start + report['length'] <= by_id[rid]['n_samples']
            # Check channel order/normalization against a raw window, independent of exporter.
            raw = np.loadtxt(by_id[ids[0]]['filepath'], delimiter=',', ndmin=2)
            expected_x = ((raw[starts[0]:starts[0]+report['length']] - detail['scaler']['mean']) / detail['scaler']['scale']).T
            np.testing.assert_allclose(x[0], expected_x, atol=1e-6, rtol=1e-6)
            for window, rid in zip(x, ids):
                key = hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()
                prior = hashes.get(key)
                if prior and prior[0] != split:
                    collisions.append([prior, [split, str(rid)]])
                hashes[key] = (split, str(rid))
            counts[split] = dict(shape=list(x.shape), records=len(set(ids)))
            split_ids[split] = set(ids)
    assert not collisions, 'Exact normalized windows shared across splits; investigate before training'
    assert not (split_ids['train'] & split_ids['validation'] or split_ids['train'] & split_ids['test'] or split_ids['test'] & split_ids['validation'])
    checks[name] = dict(passed=True, data=counts, cross_split_exact_window_collisions=0)
out = args.prepared / 'verification.json'
if out.exists():
    raise FileExistsError(out)
write_json(out, dict(checks=checks, code_sha256=digest(Path(__file__).with_name('data_pipeline.py')),
                     python=platform.python_version(), numpy=np.__version__,
                     limitations=['No model training', 'Near/shifted duplicates not fully ruled out',
                                  'Single rig/speed; no session metadata']))
print(json.dumps(checks, indent=2))
