from argparse import Namespace
import json

import numpy as np
import pytest

from experiment.pipeline import open_run, freeze_run, evaluate_run, stage
from experiment.core import save_json


def smoke_config():
    return dict(purpose='smoke_only_no_scientific_claim', profiles=['k1'], seeds=[11],
                methods=['real', 'oversample', 'simple', 'cvae'], detector_steps=4,
                generator_steps=4, eval_every=4, batch_size=16, latent_dim=32,
                detector_lr=0.001, generator_lr=0.001, beta=0.001, max_validation_fpr=0.05)


def test_smoke_cannot_be_frozen_or_access_test(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    save_json(config, smoke_config())
    run, _ = open_run(tmp_path/'run', config)
    with pytest.raises(ValueError, match='Smoke'):
        freeze_run(Namespace(run=run))
    def forbid_loader(*args, **kwargs):
        pytest.fail('Test loader reached without a final lock')
    monkeypatch.setattr('experiment.pipeline.EpsonPreparedLoader', forbid_loader)
    with pytest.raises(FileNotFoundError):
        evaluate_run(Namespace(run=run, device='cpu'))
    assert not (run/'TEST_ACCESS.json').exists()


def test_restart_skips_verified_stage_and_rejects_mutation(tmp_path):
    calls = []
    def action(folder):
        calls.append(1)
        np.save(folder/'sample.npy', np.arange(4))
    first = stage(tmp_path, 'k1', 11, 'example', action)
    assert stage(tmp_path, 'k1', 11, 'example', action) == first
    assert len(calls) == 1
    np.save(first/'sample.npy', np.arange(5))
    with pytest.raises(ValueError, match='changed'):
        stage(tmp_path, 'k1', 11, 'example', action)


def test_run_rejects_different_configuration(tmp_path):
    config = tmp_path/'config.json'
    c = smoke_config()
    save_json(config, c)
    open_run(tmp_path/'run', config)
    c['detector_steps'] += 1
    save_json(config, c)
    with pytest.raises(ValueError, match='Config differs'):
        open_run(tmp_path/'run', config)
