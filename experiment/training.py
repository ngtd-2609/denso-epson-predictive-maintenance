"""Training and inference. This module never loads the held-out test split."""
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import aggregate_recordings, metrics, save_json, select_threshold
from .models import CVAE, Detector


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def begin_resources(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def resources(start, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return dict(wall_seconds=time.perf_counter()-start,
                device=str(device), torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type == 'cuda' else None)


def tensor(x, device):
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


@torch.inference_mode()
def predict(model, x, device, batch_size=64):
    model.eval()
    return np.concatenate([model(tensor(x[i:i+batch_size], device)).softmax(1).cpu().numpy()
                           for i in range(0, len(x), batch_size)])


def assess(data, probabilities, thresholds=None, max_fpr=0.05):
    # Sum of fault classes avoids float32 cancellation in 1 - p(normal).
    scores = probabilities[:, 1:].astype(np.float64).sum(1)
    ry, rs, rid = aggregate_recordings(data.y_binary, scores, data.recording_id)
    if thresholds is None:
        thresholds = dict(window=select_threshold(data.y_binary, scores, max_fpr),
                          recording=select_threshold(ry, rs, max_fpr))
    result = dict(window=metrics(data.y_binary, scores, thresholds['window']),
                  recording=metrics(ry, rs, thresholds['recording']))
    from sklearn.metrics import confusion_matrix, f1_score
    result['six_class'] = dict(macro_f1=float(f1_score(data.y_class, probabilities.argmax(1),
                                                     labels=list(range(6)), average='macro', zero_division=0)),
                             confusion_matrix=confusion_matrix(data.y_class, probabilities.argmax(1),
                                                               labels=list(range(6))).tolist())
    result['recording_predictions'] = [dict(recording_id=str(r), y=int(y), score=float(s),
                                           prediction=int(s >= thresholds['recording']))
                                       for r, y, s in zip(rid, ry, rs)]
    record_class = np.array([data.y_class[data.recording_id == r][0] for r in rid])
    result['per_class'] = {}
    for label in range(6):
        wi, ri = data.y_class == label, record_class == label
        result['per_class'][str(label)] = dict(
            window_n=int(wi.sum()), recording_n=int(ri.sum()),
            window_alarm_rate=float((scores[wi] >= thresholds['window']).mean()) if wi.any() else None,
            recording_alarm_rate=float((rs[ri] >= thresholds['recording']).mean()) if ri.any() else None,
            interpretation='FPR' if label == 0 else 'fault recall')
    return result, thresholds


def train_detector(x, y, validation, config, seed, device, folder):
    seed_all(seed)
    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['detector_lr'])
    rng = np.random.default_rng(seed+300)
    start = begin_resources(device)
    best, best_step, logs = float('inf'), None, []
    for step in range(1, config['detector_steps']+1):
        model.train()
        ix = rng.integers(0, len(x), size=config['batch_size'])
        loss = F.cross_entropy(model(tensor(x[ix], device)), tensor(y[ix], device))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite detector loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config['eval_every'] == 0 or step == config['detector_steps']:
            prob = predict(model, validation.X, device)
            vloss = float(-np.log(np.maximum(prob[np.arange(len(prob)), validation.y_class], 1e-12)).mean())
            logs.append(dict(step=step, train_loss=float(loss.item()), validation_ce=vloss))
            if vloss < best:
                best, best_step = vloss, step
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder/'detector.pt')
            save_json(folder/'curve.json', logs)
            print(f'  detector {step}/{config["detector_steps"]} val CE={vloss:.4f}', flush=True)
    model.load_state_dict(torch.load(folder/'detector.pt', weights_only=True, map_location=device))
    prob = predict(model, validation.X, device)
    result, thresholds = assess(validation, prob, max_fpr=config['max_validation_fpr'])
    np.savez_compressed(folder/'validation_predictions.npz', probabilities=prob, y_class=validation.y_class,
                        recording_id=validation.recording_id, start_sample=validation.start_sample)
    save_json(folder/'thresholds.json', thresholds)
    save_json(folder/'validation_metrics.json', result)
    save_json(folder/'training.json', dict(seed=seed, updates=config['detector_steps'], selected_step=best_step,
                                         selection='minimum validation six-class cross entropy',
                                         train_class_counts=np.bincount(y, minlength=6).tolist(),
                                         **resources(start, device)))


def vae_loss(model, x, y, beta, eps=None):
    recon, mu, logvar = model(x, y, eps)
    mse = F.mse_loss(recon, x)
    kl = (-0.5*(1+logvar-mu.square()-logvar.exp())).mean()
    return mse+beta*kl, mse, kl


@torch.inference_mode()
def validate_vae(model, validation, device, beta, latent_dim):
    model.eval()
    ix = np.flatnonzero(validation.y_binary)
    rng = np.random.default_rng(7101)
    sums = np.zeros(3)
    for start in range(0, len(ix), 64):
        chosen = ix[start:start+64]
        eps = tensor(rng.standard_normal((len(chosen), latent_dim)).astype('float32'), device)
        losses = vae_loss(model, tensor(validation.X[chosen], device),
                          tensor(validation.y_class[chosen], device), beta, eps)
        sums += np.array([v.item() for v in losses])*len(chosen)
    return (sums/len(ix)).tolist()


def train_generator(train, validation, config, seed, device, folder):
    seed_all(seed+10000)
    model = CVAE(config['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['generator_lr'])
    rng = np.random.default_rng(seed+500)
    fault_ix = np.flatnonzero(train.y_binary)
    best, best_step, logs = float('inf'), None, []
    start = begin_resources(device)
    for step in range(1, config['generator_steps']+1):
        model.train()
        ix = rng.choice(fault_ix, size=config['batch_size'], replace=True)
        loss, mse, kl = vae_loss(model, tensor(train.X[ix], device), tensor(train.y_class[ix], device), config['beta'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite generator loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % config['eval_every'] == 0 or step == config['generator_steps']:
            val = validate_vae(model, validation, device, config['beta'], config['latent_dim'])
            logs.append(dict(step=step, train_loss=float(loss.item()), train_mse=float(mse.item()),
                             train_kl=float(kl.item()), validation_loss=val[0], validation_mse=val[1], validation_kl=val[2]))
            if val[0] < best:
                best, best_step = val[0], step
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder/'generator.pt')
            save_json(folder/'curve.json', logs)
            print(f'  generator {step}/{config["generator_steps"]} val MSE={val[1]:.4f} KL={val[2]:.4f}', flush=True)
    save_json(folder/'training.json', dict(seed=seed+10000, updates=config['generator_steps'], selected_step=best_step,
                                         selection='minimum validation MSE + beta * mean KL',
                                         fault_training_windows=len(fault_ix), **resources(start, device)))


@torch.inference_mode()
def generate(checkpoint, labels, config, seed, device):
    model = CVAE(config['latent_dim']).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(0, len(labels), 64):
        y = labels[i:i+64]
        z = rng.standard_normal((len(y), config['latent_dim'])).astype('float32')
        chunks.append(model.decode(tensor(z, device), tensor(y, device)).cpu().numpy())
    return np.concatenate(chunks)
