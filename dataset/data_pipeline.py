"""Epson-only data audit/preparation. Does not assert session independence."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE = 'https://www.epsondevice.com/sensing/en/dataset/index.html'
LABELS = {'A__D': 0, 'AW__D': 1, 'A__DW': 2, 'Avv__D': 3,
          'A__Dvv': 4, 'Avv__Dvv': 5}
NAME = re.compile(r'^trimMA342_1200rpm_(A__D|AW__D|A__DW|Avv__D|A__Dvv|Avv__Dvv)_(\d+)\.csv$')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row['recording_id'])
    return [ids for ids in groups.values() if len(ids) > 1]


def audit(root):
    rows, ignored, stats = [], [], {}
    for path in sorted(Path(root).rglob('*.csv')):
        match = NAME.fullmatch(path.name)
        if not match:
            ignored.append(str(path.resolve()))
            continue
        label = match.group(1)
        row = dict(recording_id=path.stem, filepath=str(path.resolve()),
                   class_id=LABELS[label], config_label=label, binary_label=int(LABELS[label] != 0),
                   speed_setting='unknown', rpm=1200, sampling_rate_hz=3000,
                   metadata_basis='Epson source; not measured from this CSV',
                   channels='X;Y;Z', unit='mm/s', session_id='unknown',
                   provenance='file identity only; session independence unverified',
                   source=SOURCE, source_type='measured_trimmed', bytes=path.stat().st_size,
                   checksum_sha256=digest(path), signal_sha256='', n_samples=0,
                   n_channels=0, duration_s=0, valid=False, error='')
        try:
            x = np.loadtxt(path, delimiter=',', dtype=np.float64, ndmin=2, encoding='utf-8-sig')
            row.update(n_samples=len(x), n_channels=x.shape[1], duration_s=len(x) / 3000)
            if x.shape[1] != 3 or len(x) < 2:
                raise ValueError('expected at least two rows, exactly three columns, no header')
            if not np.isfinite(x).all():
                raise ValueError('NaN/Inf values present')
            canonical = np.ascontiguousarray(x, dtype='<f8')
            row['signal_sha256'] = hashlib.sha256(canonical.tobytes()).hexdigest()
            row['valid'] = True
            stats[path.stem] = dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(),
                                   rms=np.sqrt(np.mean(x*x, axis=0)).tolist(),
                                   min=x.min(0).tolist(), max=x.max(0).tolist(),
                                   adjacent_identical_rows=int(np.all(np.diff(x, axis=0) == 0, axis=1).sum()))
        except (ValueError, OSError) as exc:
            row['error'] = str(exc)
        rows.append(row)
    if not rows:
        raise ValueError('No recognized Epson trimmed CSV files')
    if len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate recording IDs in different directories')
    report = dict(scope='local recognized Epson files only', total_records=len(rows),
                  counts_by_class=dict(Counter(str(r['class_id']) for r in rows)),
                  invalid_records=sum(not r['valid'] for r in rows), ignored_csv=ignored,
                  duplicate_file_groups=duplicate_groups(rows, 'checksum_sha256'),
                  duplicate_signal_groups=duplicate_groups(rows, 'signal_sha256'),
                  statistics=stats,
                  limitations=['Near duplicates not ruled out', 'No per-sample timestamps',
                               'Session independence unknown', 'Clipping limits not verified',
                               'Sampling rate and units from source, not measured from CSV'])
    return rows, report


def make_split(rows, train_n, val_n, test_n, seed, acknowledge_file_level=False):
    if not acknowledge_file_level:
        raise ValueError('File provenance only: acknowledge file-level evaluation, not session independence')
    if not rows or any(not r['valid'] for r in rows):
        raise ValueError('invalid audit records')
    if min(train_n, val_n, test_n) < 1:
        raise ValueError('All splits require positive record counts')
    if duplicate_groups(rows, 'signal_sha256') or duplicate_groups(rows, 'checksum_sha256'):
        raise ValueError('duplicate signals/files require provenance review before splitting')
    if len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate recording IDs')
    result = []
    for label in sorted({r['class_id'] for r in rows}):
        group = sorted([r for r in rows if r['class_id'] == label], key=lambda r: r['recording_id'])
        if len(group) != train_n + val_n + test_n:
            raise ValueError(f'insufficient or unexpected records for class {label}: {len(group)}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        for rank, index in enumerate(rng.permutation(len(group))):
            split = 'train' if rank < train_n else 'validation' if rank < train_n + val_n else 'test'
            result.append(dict(group[index], split=split, split_seed=seed, split_unit='file',
                               session_independence='unverified'))
    return sorted(result, key=lambda r: r['recording_id'])


def load_signal(row):
    return np.loadtxt(row['filepath'], delimiter=',', dtype=np.float64, ndmin=2, encoding='utf-8-sig')


def fit_scaler(rows):
    # Stable pooled population variance; each physical sample has equal weight.
    count, mean, m2 = 0, np.zeros(3), np.zeros(3)
    for row in rows:
        x = load_signal(row)
        n, batch_mean = len(x), x.mean(0)
        delta = batch_mean - mean
        m2 += np.sum((x - batch_mean)**2, axis=0) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    std = np.sqrt(m2 / count)
    return dict(mean=mean.tolist(), std=std.tolist(), scale=np.maximum(std, 1e-8).tolist(),
                n_samples=count, fit_recording_ids=[r['recording_id'] for r in rows],
                weighting='all samples in selected train records, including unused tail', ddof=0)


def materialize(rows, selected_ids, output, length):
    output.mkdir()
    selected = [r for r in rows if r['recording_id'] in selected_ids]
    scaler = fit_scaler(selected)
    write_json(output / 'scaler.json', scaler)
    windows, counts = [], {}
    for split in ('train', 'validation', 'test'):
        xs, ys, record_ids, starts = [], [], [], []
        for row in rows:
            if row['split'] != split or (split == 'train' and row['recording_id'] not in selected_ids):
                continue
            x = load_signal(row)
            n = len(x) // length
            if not n:
                raise ValueError(f'Record shorter than window: {row["recording_id"]}')
            # One offset shared by X/Y/Z; no filtering, phase shift or resampling.
            values = ((x[:n*length] - scaler['mean']) / scaler['scale']).astype(np.float32)
            xs.append(values.reshape(n, length, 3).transpose(0, 2, 1))
            ys.extend([row['class_id']] * n)
            record_ids.extend([row['recording_id']] * n)
            starts.extend(range(0, n * length, length))
            for start in range(0, n * length, length):
                windows.append(dict(profile=output.name, split=split, recording_id=row['recording_id'],
                                    window_id=f'{row["recording_id"]}:{start}', start_sample=start,
                                    end_sample_exclusive=start + length, class_id=row['class_id'],
                                    binary_label=row['binary_label'], fs_hz=3000, rpm=1200))
        if not xs:
            raise ValueError(f'Empty {split}')
        np.savez(output / f'{split}.npz', X=np.concatenate(xs), y_class=np.array(ys, dtype=np.int64),
                 y_binary=(np.array(ys) != 0).astype(np.int64), recording_id=np.array(record_ids),
                 start_sample=np.array(starts, dtype=np.int64))
        counts[split] = len(ys)
    write_csv(output / 'windows.csv', windows)
    return dict(train_ids=sorted(selected_ids), scaler=scaler, window_counts=counts)


def prepare(rows, output, budgets, length, seed):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new run: {output}')
    if length < 2 or not budgets or min(budgets) < 1:
        raise ValueError('Invalid window length or scarcity budget')
    ids = [r['recording_id'] for r in rows]
    if len(ids) != len(set(ids)) or any(r.get('split') not in ('train', 'validation', 'test') for r in rows):
        raise ValueError('Invalid/overlapping split records')
    if any(not r['valid'] for r in rows) or duplicate_groups(rows, 'signal_sha256'):
        raise ValueError('invalid or duplicate records')
    for row in rows:
        if digest(row['filepath']) != row['checksum_sha256']:
            raise ValueError(f'Source changed after audit: {row["recording_id"]}')
        if row['n_samples'] < length:
            raise ValueError('Window longer than record')
    normal = sorted(r['recording_id'] for r in rows if r['split'] == 'train' and r['class_id'] == 0)
    if not normal:
        raise ValueError('No normal training records')
    choices = {}
    for label in sorted({r['class_id'] for r in rows} - {0}):
        pool = sorted(r['recording_id'] for r in rows if r['split'] == 'train' and r['class_id'] == label)
        if len(pool) < max(budgets):
            raise ValueError(f'insufficient fault train records for class {label}')
        choices[label] = np.random.default_rng(np.random.SeedSequence([seed, label])).permutation(pool).tolist()
    if not choices:
        raise ValueError('No fault classes')
    output.mkdir(parents=True)
    result = dict(status='complete', source='Epson trimmed', split_unit='file',
                  limitation='Session independence unverified; single rig/speed',
                  length=length, stride=length, fs_hz=3000, channels=['X', 'Y', 'Z'],
                  unit_raw='mm/s', unit_X='dimensionless z-score', seed=seed, budgets={},
                  input_checksums={r['recording_id']: r['checksum_sha256'] for r in rows},
                  unused_tail_samples={r['recording_id']: r['n_samples'] % length for r in rows},
                  labeled_validation_records=sum(r['split'] == 'validation' and r['class_id'] != 0 for r in rows))
    for k in sorted(set(budgets)):
        fault_ids = [rid for pool in choices.values() for rid in pool[:k]]
        detail = materialize(rows, set(normal + fault_ids), output / f'k{k}', length)
        detail['fault_train_ids'] = sorted(fault_ids)
        result['budgets'][str(k)] = detail
    result['normal_only'] = materialize(rows, set(normal), output / 'normal_only', length)
    # Completion marker is written last; partial folders must not be used as ready data.
    write_json(output / 'preparation_report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    a = subs.add_parser('audit')
    a.add_argument('--raw', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    s = subs.add_parser('split')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--acknowledge-file-level', action='store_true')
    p = subs.add_parser('prepare')
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--budgets', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--length', type=int, default=3000)
    args = parser.parse_args()
    if args.command == 'audit':
        if args.out.exists():
            raise FileExistsError(args.out)
        rows, report = audit(args.raw)
        args.out.mkdir(parents=True)
        write_json(args.out / 'manifest.json', rows)
        write_csv(args.out / 'public_manifest.csv', rows)
        write_json(args.out / 'audit_report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'statistics'}, indent=2))
    elif args.command == 'split':
        if args.out.exists():
            raise FileExistsError(args.out)
        rows = json.loads(args.manifest.read_text(encoding='utf-8'))
        if {r['class_id'] for r in rows} != set(range(6)):
            raise ValueError('Need all six Epson classes; sample-only audit cannot create research split')
        result = make_split(rows, 8, 4, 4, args.seed, args.acknowledge_file_level)
        args.out.mkdir(parents=True)
        write_json(args.out / 'split.json', result)
        write_csv(args.out / 'split.csv', result)
        print(dict(Counter(r['split'] for r in result)))
    else:
        rows = json.loads(args.split.read_text(encoding='utf-8'))
        result = prepare(rows, args.out, args.budgets, args.length, args.seed)
        print(json.dumps({k: v['window_counts'] for k, v in result['budgets'].items()}))


if __name__ == '__main__':
    main()
