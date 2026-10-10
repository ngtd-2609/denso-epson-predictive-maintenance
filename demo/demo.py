"""Offline generation demo. Frozen test scores are displayed, never recomputed."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from model_definition import CVAE

def main():
    ap=argparse.ArgumentParser(description='Epson cVAE research demo; outputs are synthetic, not measurements.')
    ap.add_argument('--class-id',type=int,choices=range(1,6),default=1)
    ap.add_argument('--samples',type=int,default=5)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not 1 <= a.samples <= 100: ap.error('--samples must be 1..100')
    if not 0 <= a.seed < 2**32: ap.error('--seed must be 0..4294967295')
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        ap.error('Output must be a new or empty directory; existing files are protected.')
    base=Path(__file__).resolve().parent
    m=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    for filename,key in [('generator.pt','generator_sha256'),('model_definition.py','model_definition_sha256')]:
        if hashlib.sha256((base/filename).read_bytes()).hexdigest()!=m[key]:
            raise ValueError('Integrity mismatch: '+filename)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    model=CVAE(m['latent_dim']).eval()
    model.load_state_dict(torch.load(base/'generator.pt',map_location='cpu',weights_only=True))
    rng=torch.Generator(device='cpu').manual_seed(a.seed)
    labels=torch.full((a.samples,),a.class_id,dtype=torch.long)
    with torch.inference_mode():
        x=model.decode(torch.randn(a.samples,m['latent_dim'],generator=rng),labels).numpy()
    raw=x*np.array(m['scale'])[None,:,None]+np.array(m['mean'])[None,:,None]
    if x.shape!=(a.samples,3,3000) or not np.isfinite(raw).all(): raise ValueError('Invalid generated data')
    rms=np.sqrt((raw**2).mean(-1))
    ratio=rms.mean(0)/np.array(m['train_mean_window_rms'][str(a.class_id)])
    warning=bool(((ratio<0.5)|(ratio>2)).any())
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'synthetic_samples.npz',X=x,X_raw_mm_s=raw,y_class=labels.numpy(),
                        is_synthetic=np.ones(a.samples,dtype=bool),sampling_rate_hz=3000)
    np.savetxt(a.out/'first_sample.csv',np.column_stack([np.arange(3000)/3000,raw[0].T]),
               delimiter=',',header='time_s,X_mm_s,Y_mm_s,Z_mm_s',comments='')
    report=dict(class_id=a.class_id,samples=a.samples,seed=a.seed,device='cpu',
                generator_sha256=m['generator_sha256'],source='N(0,I) prior; synthetic research data',
                rms_ratio_to_train_xyz=ratio.tolist(),rms_warning=warning,
                warning_rule='Outside [0.5,2] in any channel; diagnostic only, no automatic filtering',
                scope='Generation only. No retraining, no fresh test evaluation, no physical certification.')
    (a.out/'generation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GENERATED:',a.samples,'windows; shape',x.shape,'; unit mm/s after inverse transform')
    print('RMS ratio XYZ:',np.round(ratio,4),'WARNING:',warning)
    print('\nFROZEN HELD-OUT RESULTS: means over 3 seeds, same 24 test recordings; NOT a test of this batch.')
    rows=list(csv.DictReader((base/'metrics_frozen.csv').open(encoding='utf-8-sig')))
    for profile in ['k1','k4']:
        for method in ['real','oversample','simple','cvae']:
            group=[r for r in rows if r['profile']==profile and r['method']==method and r['split']=='test' and r['level']=='recording']
            if len(group)!=3: raise ValueError('Expected 3 frozen seeds')
            print(f"{profile:2} {method:10} Recall={np.mean([float(r['recall']) for r in group])*100:5.1f}% FPR={np.mean([float(r['fpr']) for r in group])*100:5.1f}%")
    print('\nOutput:',a.out.resolve())
    print('Research prototype. Current cVAE underperforms Simple; no claim about unseen faults or RUL.')

if __name__=='__main__': main()
