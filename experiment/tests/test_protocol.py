import numpy as np
import pytest

from experiment.core import select_threshold, metrics, aggregate_recordings, augmentation


def test_threshold_with_tied_normal_scores_respects_false_positive_budget():
    y = np.array([0] * 20 + [1, 1])
    scores = np.array([0.2] * 18 + [0.7, 0.7, 0.8, 0.9])
    threshold = select_threshold(y, scores, max_fpr=0.05)
    result = metrics(y, scores, threshold)
    assert result['fpr'] == 0
    assert result['recall'] == 1


def test_recording_metrics_count_groups_instead_of_windows():
    y, scores, ids = aggregate_recordings(
        np.array([0, 0, 1]), np.array([0.1, 0.3, 0.9]), np.array(['a', 'a', 'b']))
    np.testing.assert_allclose(scores, [0.2, 0.9])
    assert ids.tolist() == ['a', 'b']
    assert metrics(y, scores, 0.5)['confusion_matrix'] == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match='labels'):
        aggregate_recordings(np.array([0, 1]), np.array([0.1, 0.2]), np.array(['a', 'a']))


def test_augmentation_preserves_interchannel_phase_and_does_not_mutate_input():
    x = np.arange(24, dtype=np.float32).reshape(2, 1, 12) + 1
    x = np.concatenate([x, 2*x, -x], axis=1)
    before = x.copy()
    result = augmentation(x, np.random.default_rng(42))
    np.testing.assert_allclose(result[:, 1], 2*result[:, 0])
    np.testing.assert_allclose(result[:, 2], -result[:, 0])
    np.testing.assert_array_equal(x, before)


def test_metrics_report_undefined_rates_and_reject_invalid_predictions():
    assert metrics(np.ones(2, dtype=int), np.array([0.2, 0.8]), 0.5)['fpr'] is None
    with pytest.raises(ValueError):
        select_threshold(np.ones(2), np.array([0.2, 0.8]), 0.05)
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0.3, np.nan]), 0.5)
