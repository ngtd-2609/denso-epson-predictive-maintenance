from types import SimpleNamespace

import numpy as np

from experiment.quality import quality_report
from experiment.training import assess


def test_fault_recall_per_class_uses_fixed_threshold():
    data = SimpleNamespace(y_class=np.array([0, 0, 1, 2]), y_binary=np.array([0, 0, 1, 1]),
                           recording_id=np.array(['n', 'n', 'a', 'b']))
    probabilities = np.zeros((4, 6))
    probabilities[:, 0] = [0.9, 0.7, 0.2, 0.8]
    probabilities[:, 1] = 1-probabilities[:, 0]
    result, thresholds = assess(data, probabilities, dict(window=0.5, recording=0.5))
    assert result['per_class']['0']['recording_alarm_rate'] == 0
    assert result['per_class']['1']['recording_alarm_rate'] == 1
    assert result['per_class']['2']['recording_alarm_rate'] == 0
    assert result['per_class']['3']['recording_alarm_rate'] is None
    assert thresholds == dict(window=0.5, recording=0.5)


def test_synthetic_diagnostics_detect_copied_waveforms():
    x = np.random.default_rng(1).normal(size=(5, 3, 512)).astype('float32')
    y = np.arange(1, 6)
    result = quality_report(x, y, x.copy(), y, np.zeros(3), np.ones(3))
    for row in result['classes'].values():
        assert row['exact_training_duplicates'] == 1
        assert row['near_training_windows_relative_l2_lt_005'] == 1
        np.testing.assert_allclose(row['rms_ratio'], [1, 1, 1])
