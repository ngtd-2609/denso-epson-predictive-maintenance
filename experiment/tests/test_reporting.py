from argparse import Namespace

from experiment.core import save_json
from experiment.tests.test_gates import smoke_config
from experiment import reporting


def test_report_reads_training_seed_once_and_keeps_observed_metrics(tmp_path, monkeypatch):
    folder = tmp_path/'model'
    folder.mkdir()
    save_json(folder/'training.json', dict(seed=11, updates=4, wall_seconds=1.0))
    value = dict(n=24, confusion_matrix=[[4, 0], [5, 15]], recall=0.75, fpr=0.0,
                 macro_f1=0.8, average_precision=0.9, threshold=0.5)
    save_json(folder/'validation_metrics.json', dict(window=value, recording=value))
    def get_stage(run, profile, seed, method):
        if method == 'real':
            return folder
        raise RuntimeError('not yet run')
    monkeypatch.setattr(reporting, 'open_run', lambda run: (tmp_path, smoke_config()))
    monkeypatch.setattr(reporting, 'completed', get_stage)
    # Rendering is tested by the actual smoke run; this regression checks report data flow.
    monkeypatch.setattr(reporting, 'save_figure', lambda fig, base: reporting.plt.close(fig))
    reporting.report_run(Namespace(run=tmp_path))
    assert '0.750' in (tmp_path/'report/BAO_CAO.md').read_text(encoding='utf-8')
    assert 'validation' in (tmp_path/'report/metrics.csv').read_text(encoding='utf-8-sig')
