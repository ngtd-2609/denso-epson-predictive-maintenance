"""Frozen Epson experiment. Stages are restartable; test is a separate locked command."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

from person1_data.loader_v1 import EpsonPreparedLoader
from .core import augmentation, file_hash, load_verified, save_json
from .models import Detector
from .quality import quality_report
from .training import assess, generate, predict, train_detector, train_generator

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['real', 'oversample', 'simple', 'cvae']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    return {p.name: file_hash(p) for p in sorted((ROOT/'experiment').glob('*.py'))}


def device_for(name):
    torch.set_num_threads(4)
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use .venv-train/Scripts/python.exe; run doctor first.')
    return torch.device(name)


def validate_config(c):
    if c['methods'] != METHODS or not c['profiles'] or not set(c['profiles']) <= {'k1', 'k2', 'k4'}:
        raise ValueError('Use four prescribed methods and valid scarcity profiles')
    if len(set(c['profiles'])) != len(c['profiles']) or len(set(c['seeds'])) != len(c['seeds']) or not c['seeds']:
        raise ValueError('Profiles and seeds must be nonempty and unique')
    for key in ['detector_steps', 'generator_steps', 'eval_every', 'batch_size', 'latent_dim']:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'Invalid {key}')
    if not 0 <= c['max_validation_fpr'] < 1 or c['beta'] < 0:
        raise ValueError('Invalid FPR or beta')
    if c['detector_lr'] <= 0 or c['generator_lr'] <= 0:
        raise ValueError('Learning rates must be positive')


def protocol_for(config):
    return dict(config=config, source_sha256=sources(),
                data_contract_sha256=file_hash(ROOT/'person1_data/results/qa_v2/data_contract.json'),
                prepared_manifest_sha256=file_hash(ROOT/'person1_data/prepared_arrays_manifest.json'),
                detector='Conv1d(3,16,32,64;k11,s5), ReLU, global mean, Linear(64,6)',
                generator='conditional waveform VAE, signed linear output, N(0,I) prior',
                threshold='smallest validation-normal threshold at empirical FPR <= configured limit; predict >=',
                recording_aggregation='mean binary fault probability; separate validation recording threshold',
                checkpoint_selection='minimum validation CE for detector; minimum validation MSE+beta*KL for cVAE',
                balancing='B/C/D add fault windows to match 752 normal windows/class; A real only',
                simple_augmentation='joint XYZ amplitude U(0.9,1.1), joint circular shift integer U(-150,150)',
                test_access='only evaluate command after explicit freeze command',
                created_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def open_run(run, config_path=None):
    run = Path(run).resolve()
    protocol_path = run/'protocol.json'
    if not protocol_path.exists():
        if config_path is None:
            raise FileNotFoundError('Run has no protocol.json')
        if run.exists() and any(run.iterdir()):
            raise ValueError('Refusing a nonempty directory without a protocol')
        c = read_json(config_path)
        validate_config(c)
        run.mkdir(parents=True, exist_ok=True)
        save_json(protocol_path, protocol_for(c))
        snapshot = run/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in (ROOT/'experiment').glob('*.py'):
            shutil.copy2(source, snapshot/source.name)
    p = read_json(protocol_path)
    if p['source_sha256'] != sources():
        raise ValueError('Source differs from frozen run. Restore source or create a new run; never mix versions.')
    if config_path and p['config'] != read_json(config_path):
        raise ValueError('Config differs from frozen run')
    for key, path in [('data_contract_sha256', ROOT/'person1_data/results/qa_v2/data_contract.json'),
                      ('prepared_manifest_sha256', ROOT/'person1_data/prepared_arrays_manifest.json')]:
        if p[key] != file_hash(path):
            raise ValueError('Frozen data provenance changed')
    return run, p['config']


def check_stage(folder):
    marker = folder/'complete.json'
    if not marker.exists():
        return False
    for name, sha in read_json(marker)['files'].items():
        if file_hash(folder/name) != sha:
            raise ValueError(f'Completed artifact changed: {folder/name}')
    return True


def stage(run, profile, seed, name, operation):
    parent = run/profile/f'seed_{seed}'/name
    parent.mkdir(parents=True, exist_ok=True)
    for folder in sorted(parent.glob('attempt_*')):
        if check_stage(folder):
            print(f'SKIP verified {profile}/{seed}/{name}', flush=True)
            return folder
    folder = parent/f'attempt_{len(list(parent.glob("attempt_*")))+1:03d}'
    folder.mkdir()
    print(f'START {profile}/{seed}/{name}', flush=True)
    operation(folder)
    save_json(folder/'complete.json', dict(files={p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}))
    return folder


def completed(run, profile, seed, name):
    for p in sorted((run/profile/f'seed_{seed}'/name).glob('attempt_*')):
        if check_stage(p):
            return p
    raise RuntimeError(f'Incomplete stage: {profile}/{seed}/{name}')


def augment_data(train, method, config, seed, checkpoint, device, folder, scaler):
    source_rng = np.random.default_rng(seed+700)
    counts = np.bincount(train.y_class, minlength=6)
    source_indices = np.concatenate([source_rng.choice(np.flatnonzero(train.y_class == label),
                                                       size=int(counts[0]-counts[label]), replace=True)
                                     for label in range(1, 6)])
    y = train.y_class[source_indices]
    if method == 'oversample':
        x = train.X[source_indices].copy()
    elif method == 'simple':
        x = augmentation(train.X[source_indices], np.random.default_rng(seed+800))
    else:
        x = generate(checkpoint, y, config, seed+900, device)
    if not np.isfinite(x).all() or x.shape != (len(y), 3, 3000):
        raise ValueError('Invalid generated batch')
    source_id = train.recording_id[source_indices] if method != 'cvae' else np.full(len(y), '')
    synthetic_id = np.array([f'{method}_s{seed}_{i:05d}' for i in range(len(y))])
    np.savez_compressed(folder/'added.npz', X=x, y_class=y, y_binary=np.ones(len(y), dtype=np.int64),
                        synthetic_id=synthetic_id, source_recording_id=source_id)
    mean, scale = np.array(scaler['mean']), np.array(scaler['scale'])
    if method == 'cvae':
        save_json(folder/'quality.json', quality_report(train.X, train.y_class, x, y, mean, scale))
    save_json(folder/'dataset.json', dict(method=method, added_class_counts=np.bincount(y, minlength=6).tolist(),
                                        real_train_recording_ids=np.unique(train.recording_id).tolist(),
                                        synthetic_ids_are_not_independent_recordings=True,
                                        unit='dimensionless z-score', inverse_transform='X_raw = X * scale + mean',
                                        mean=mean.tolist(), scale=scale.tolist()))


def train_run(args):
    device = device_for(args.device)
    run, c = open_run(args.run, args.config)
    if (run/'FINAL_LOCK.json').exists():
        raise ValueError('Run is locked for final evaluation; training is disabled')
    loader = EpsonPreparedLoader()
    for profile in c['profiles']:
        train = load_verified(loader, profile, 'train')
        validation = load_verified(loader, profile, 'validation')
        scaler = loader.get_scaler(profile)
        for seed in c['seeds']:
            generator = stage(run, profile, seed, 'generator', lambda f: train_generator(train, validation, c, seed, device, f))
            for method in c['methods']:
                if method == 'real':
                    x, y = train.X, train.y_class
                else:
                    added = stage(run, profile, seed, method+'_data',
                                  lambda f: augment_data(train, method, c, seed, generator/'generator.pt', device, f, scaler))
                    with np.load(added/'added.npz', allow_pickle=False) as data:
                        x = np.concatenate([train.X, data['X']])
                        y = np.concatenate([train.y_class, data['y_class']])
                stage(run, profile, seed, method, lambda f: train_detector(x, y, validation, c, seed, device, f))
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    save_json(run/'environment.json', environment(device))
    print('TRAINING COMPLETE. Review validation and quality; freeze before any test evaluation.', flush=True)


def environment(device):
    import scipy
    import sklearn
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                cuda_runtime=torch.version.cuda, device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                cudnn_deterministic=True, bitwise_reproducibility_across_hardware_guaranteed=False)


def freeze_run(args):
    run, c = open_run(args.run)
    if c['purpose'] != 'main_frozen_comparison':
        raise ValueError('Smoke runs cannot be evaluated on test')
    paths = [run/'protocol.json']
    for profile in c['profiles']:
        for seed in c['seeds']:
            for name in ['generator']+c['methods']+[m+'_data' for m in c['methods'] if m != 'real']:
                folder = completed(run, profile, seed, name)
                paths.extend(p for p in folder.iterdir() if p.is_file())
    lock = dict(files={str(p.relative_to(run)).replace('\\', '/'): file_hash(p) for p in paths},
                rule='No model, data processing, metric, threshold or hyperparameter changes after test access')
    if (run/'FINAL_LOCK.json').exists():
        if read_json(run/'FINAL_LOCK.json') != lock:
            raise ValueError('Existing final lock differs')
    else:
        save_json(run/'FINAL_LOCK.json', lock)
    print('FINAL PROTOCOL LOCKED. Test evaluation can now run.', flush=True)


def verify_lock(run):
    lock = read_json(run/'FINAL_LOCK.json')
    for name, sha in lock['files'].items():
        if file_hash(run/name) != sha:
            raise ValueError(f'Final locked file changed: {name}')


def evaluate_run(args):
    device = device_for(args.device)
    run, c = open_run(args.run)
    verify_lock(run)
    loader = EpsonPreparedLoader()
    # A durable audit note is written before the first model access to test.
    if not (run/'TEST_ACCESS.json').exists():
        save_json(run/'TEST_ACCESS.json', dict(started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                             final_lock_sha256=file_hash(run/'FINAL_LOCK.json')))
    for profile in c['profiles']:
        test = load_verified(loader, profile, 'test')
        for seed in c['seeds']:
            for method in c['methods']:
                folder = completed(run, profile, seed, method)
                def evaluate(out):
                    model = Detector().to(device)
                    model.load_state_dict(torch.load(folder/'detector.pt', weights_only=True, map_location=device))
                    prob = predict(model, test.X, device)
                    result, _ = assess(test, prob, read_json(folder/'thresholds.json'))
                    save_json(out/'metrics.json', result)
                    np.savez_compressed(out/'predictions.npz', probabilities=prob, y_class=test.y_class,
                                        recording_id=test.recording_id, start_sample=test.start_sample)
                stage(run, profile, seed, method+'_test', evaluate)
    print('FINAL TEST EVALUATION COMPLETE. Do not tune using these results.', flush=True)


def export_dataset(args):
    run, c = open_run(args.run)
    if args.profile not in c['profiles'] or args.seed not in c['seeds']:
        raise ValueError('Profile/seed not in this run')
    folder = completed(run, args.profile, args.seed, args.method+'_data')
    train = load_verified(EpsonPreparedLoader(), args.profile, 'train')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'added.npz', allow_pickle=False) as data:
        n = len(data['X'])
        all_x = np.concatenate([train.X, data['X']])
        metadata = read_json(folder/'dataset.json')
        raw = all_x*np.array(metadata['scale'], dtype=np.float32)[None, :, None]
        raw += np.array(metadata['mean'], dtype=np.float32)[None, :, None]
        np.savez_compressed(out, X=all_x, X_raw_mm_s=raw,
                            y_class=np.concatenate([train.y_class, data['y_class']]),
                            y_binary=np.concatenate([train.y_binary, data['y_binary']]),
                            is_added=np.r_[np.zeros(len(train), dtype=bool), np.ones(n, dtype=bool)],
                            source_recording_id=np.r_[train.recording_id, data['source_recording_id']],
                            start_sample=np.r_[train.start_sample, np.full(n, -1)],
                            synthetic_id=np.r_[np.full(len(train), ''), data['synthetic_id']])
    save_json(out.with_suffix('.json'), dict(profile=args.profile, seed=args.seed, method=args.method,
                                           file_sha256=file_hash(out), unit='dimensionless z-score',
                                           note='Training only. Added windows are not independent recordings.',
                                           scaler=read_json(folder/'dataset.json')))
    print(f'EXPORTED {out}', flush=True)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    for name in ['train', 'freeze', 'evaluate', 'report', 'export', 'demo']:
        p = sub.add_parser(name)
        p.add_argument('--run', required=True)
        if name in ['train', 'evaluate', 'demo']:
            p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        if name == 'train':
            p.add_argument('--config', required=True)
        if name in ['export', 'demo']:
            p.add_argument('--profile', default='k1')
            p.add_argument('--seed', type=int, default=11)
            p.add_argument('--out', required=True)
        if name == 'export':
            p.add_argument('--method', choices=METHODS[1:], default='cvae')
        if name == 'demo':
            p.add_argument('--class-id', type=int, choices=range(1, 6), default=1)
            p.add_argument('--sample-seed', type=int, default=2026)
    args = parser.parse_args()
    if args.command == 'doctor':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(json.dumps(environment(device), indent=2))
        if device.type == 'cuda':
            print('CUDA tensor check:', (torch.ones(4, device=device)*2).sum().item())
        return
    if args.command == 'report':
        from .reporting import report_run
        report_run(args)
    elif args.command == 'demo':
        from .reporting import demo
        demo(args)
    else:
        {'train': train_run, 'freeze': freeze_run, 'evaluate': evaluate_run, 'export': export_dataset}[args.command](args)
