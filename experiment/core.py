"""Pure protocol operations; thresholds are fitted on validation only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


def _check(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty matching one-dimensional labels/scores')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid binary labels or nonfinite scores')
    return y, scores


def select_threshold(y, scores, max_fpr=0.05):
    y, scores = _check(y, scores)
    normal = np.sort(scores[y == 0])[::-1]
    if not len(normal) or not 0 <= max_fpr < 1:
        raise ValueError('Need normal validation examples and 0 <= max_fpr < 1')
    # >= comparison: move above the first forbidden normal, including all ties.
    allowed = int(np.floor(max_fpr * len(normal)))
    return float(np.nextafter(normal[allowed], np.inf))


def metrics(y, scores, threshold):
    y, scores = _check(y, scores)
    if not np.isfinite(threshold):
        raise ValueError('Nonfinite threshold')
    pred = scores >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return dict(n=len(y), confusion_matrix=cm.tolist(),
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                macro_f1=float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
                average_precision=float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
                threshold=float(threshold))


def aggregate_recordings(y, scores, recording_id):
    y, scores = _check(y, scores)
    recording_id = np.asarray(recording_id)
    if recording_id.shape != y.shape:
        raise ValueError('Mismatching recording IDs')
    ids, labels, means = np.unique(recording_id), [], []
    for rid in ids:
        ix = recording_id == rid
        if len(np.unique(y[ix])) != 1:
            raise ValueError('Conflicting labels in one recording')
        labels.append(y[ix][0])
        means.append(scores[ix].mean())
    return np.array(labels), np.array(means), ids


def augmentation(x, rng):
    """Common XYZ scale and circular shift retain channel phase relationships."""
    result = np.empty_like(x)
    for i, window in enumerate(x):
        shift = int(rng.integers(-min(150, x.shape[-1]), min(150, x.shape[-1])+1))
        result[i] = np.roll(window, shift, axis=-1) * rng.uniform(0.9, 1.1)
    return result


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_verified(loader, profile, split):
    """Check frozen array hashes as well as the official loader's contract."""
    data = loader.load(profile, split)
    manifest = json.loads((loader.root / 'prepared_arrays_manifest.json').read_text(encoding='utf-8'))
    expected = manifest['files'][f'{profile}/{split}.npz']['arrays']
    for key, signature in expected.items():
        value = getattr(data, key)
        value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<')))
        actual = dict(shape=list(value.shape), dtype=value.dtype.str,
                      sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if actual != signature:
            raise ValueError(f'Frozen array mismatch: {profile}/{split}/{key}')
    return data
