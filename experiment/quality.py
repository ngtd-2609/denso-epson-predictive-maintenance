"""Training-only descriptive diagnostics, not a synthetic-data acceptance filter."""
import hashlib

import numpy as np
from scipy.signal import welch
from sklearn.metrics import pairwise_distances


def signal_features(x):
    rms = np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1))
    centered = x-x.mean(axis=-1, keepdims=True)
    norms = np.sqrt((centered**2).sum(-1))
    corr = np.stack([(centered[:, a]*centered[:, b]).sum(-1)/np.maximum(norms[:, a]*norms[:, b], 1e-20)
                     for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    frequency, psd = welch(x, fs=3000, nperseg=512, axis=-1)
    bands = np.stack([psd[..., (frequency >= lo) & (frequency < hi)].sum(-1)
                      for lo, hi in [(0, 50), (50, 200), (200, 500), (500, 1501)]], axis=-1)
    fractions = bands/np.maximum(bands.sum(-1, keepdims=True), 1e-30)
    features = np.concatenate([np.log(np.maximum(rms, 1e-15)), corr, fractions.reshape(len(x), -1)], axis=1)
    return rms, corr, psd, features


def quality_report(real, real_y, synthetic, synthetic_y, mean, scale):
    result = {'scope': 'allowed training only; raw mm/s; descriptive, no rejection/filtering', 'classes': {}}
    for label in range(1, 6):
        r = real[real_y == label]
        s = synthetic[synthetic_y == label]
        raw_r = r*scale[None, :, None]+mean[None, :, None]
        raw_s = s*scale[None, :, None]+mean[None, :, None]
        rr, rc, rp, rf = signal_features(raw_r)
        sr, sc, sp, sf = signal_features(raw_s)
        # Standardization belongs only to this diagnostic feature distance.
        spread = np.maximum(rf.std(0), 1e-5)
        nearest = pairwise_distances((sf-rf.mean(0))/spread, (rf-rf.mean(0))/spread).min(1)
        relative_distances, nearest_indices = [], []
        rflat = r.reshape(len(r), -1)
        rnorm = np.maximum(np.linalg.norm(rflat, axis=1), 1e-12)
        for begin in range(0, len(s), 64):
            distance = pairwise_distances(s[begin:begin+64].reshape(-1, rflat.shape[1]), rflat)/rnorm[None, :]
            nearest_indices.extend(distance.argmin(1).tolist())
            relative_distances.extend(distance.min(1).tolist())
        closest = np.argsort(relative_distances)[:5]
        rhashes = {hashlib.sha256(v.tobytes()).digest() for v in r}
        shashes = [hashlib.sha256(v.tobytes()).digest() for v in s]
        ratio = sr.mean(0)/np.maximum(rr.mean(0), 1e-15)
        rp_mean, sp_mean = rp.mean(0), sp.mean(0)
        rp_norm = rp_mean/np.maximum(rp_mean.sum(-1, keepdims=True), 1e-30)
        sp_norm = sp_mean/np.maximum(sp_mean.sum(-1, keepdims=True), 1e-30)
        result['classes'][str(label)] = dict(
            real_n=len(r), synthetic_n=len(s), rms_real=rr.mean(0).tolist(), rms_synthetic=sr.mean(0).tolist(),
            rms_ratio=ratio.tolist(), channel_correlation_real=rc.mean(0).tolist(),
            channel_correlation_synthetic=sc.mean(0).tolist(),
            normalized_psd_l1=np.abs(rp_norm-sp_norm).sum(-1).tolist(),
            feature_spread_ratio=float(np.linalg.norm(sf.std(0))/max(np.linalg.norm(rf.std(0)), 1e-12)),
            nearest_training_feature_distance_median=float(np.median(nearest)),
            exact_training_duplicates=sum(h in rhashes for h in shashes),
            raw_max_abs_real=np.max(np.abs(raw_r), axis=(0, 2)).tolist(),
            raw_max_abs_synthetic=np.max(np.abs(raw_s), axis=(0, 2)).tolist(),
            fraction_beyond_training_channel_extrema=(np.abs(raw_s) > np.max(np.abs(raw_r), axis=(0, 2))[None, :, None]).mean((0, 2)).tolist(),
            relative_waveform_distance_median=float(np.median(relative_distances)),
            near_training_windows_relative_l2_lt_005=int(np.sum(np.array(relative_distances) < 0.05)),
            nearest_waveform_examples=[dict(synthetic_class_index=int(i), real_class_index=nearest_indices[i],
                                            relative_l2=relative_distances[i]) for i in closest],
            unique_synthetic_windows=len(set(shashes)),
            rms_warning=bool(np.any((ratio < 0.5) | (ratio > 2))))
    result['interpretation'] = ('RMS warning uses descriptive 0.5–2 range, not a physical acceptance criterion. '
                                'Relative waveform L2 uses synchronized unshifted z-score windows; shifted copies may escape detection. '
                                'Training extrema are a diagnostic reference, not sensor clipping limits. '
                                'Good reconstruction does not guarantee good prior-generated signals.')
    return result
