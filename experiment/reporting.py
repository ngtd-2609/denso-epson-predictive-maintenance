"""Report observed values only. Single-panel figures preserve every completed seed."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from dataset.loader_v1 import EpsonPreparedLoader
from .core import load_verified, save_json
from .pipeline import METHODS, completed, device_for, open_run, read_json
from .training import generate

DISPLAY = dict(real='Real', oversample='Oversample', simple='Simple', cvae='cVAE')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False})


def save_figure(fig, base):
    fig.tight_layout()
    fig.canvas.draw()
    assert len(fig.axes) == 1, 'This exporter is for single-panel figures only'
    save_json(base.with_suffix('.alignment.json'), dict(status='NOT APPLICABLE', axes_count=1))
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'), dpi=300)
    plt.close(fig)


def report_run(args):
    run, c = open_run(args.run)
    out = run/'report'
    out.mkdir(exist_ok=True)
    rows, resources, quality = [], [], []
    for profile in c['profiles']:
        for seed in c['seeds']:
            for method in METHODS:
                try:
                    folder = completed(run, profile, seed, method)
                except RuntimeError:
                    continue
                resource = read_json(folder/'training.json')
                resources.append(dict(profile=profile, method=method, **resource))
                for split in ['validation', 'test']:
                    if split == 'validation':
                        result = read_json(folder/'validation_metrics.json')
                    else:
                        try:
                            result = read_json(completed(run, profile, seed, method+'_test')/'metrics.json')
                        except RuntimeError:
                            continue
                    for level in ['window', 'recording']:
                        row = dict(profile=profile, seed=seed, method=method, split=split, level=level)
                        row.update({k: v for k, v in result[level].items() if k != 'confusion_matrix'})
                        tn, fp, fn, tp = np.array(result[level]['confusion_matrix']).ravel().tolist()
                        row.update(tn=tn, fp=fp, fn=fn, tp=tp)
                        rows.append(row)
            try:
                q = read_json(completed(run, profile, seed, 'cvae_data')/'quality.json')
                quality.append(dict(profile=profile, seed=seed, **q))
            except RuntimeError:
                pass
    if not rows:
        raise RuntimeError('No completed detector result to report')
    with (out/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out/'resources.json', resources)
    save_json(out/'synthetic_quality.json', quality)
    split = 'test' if any(r['split'] == 'test' for r in rows) else 'validation'
    total = len(c['profiles'])*len(c['seeds'])*len(METHODS)
    count = sum(r['split'] == split and r['level'] == 'recording' for r in rows)
    lines = ['# Báo cáo thí nghiệm Epson', '',
             f"Mục đích cấu hình: `{c['purpose']}`. Đã có {count}/{total} kết quả {split}.", '',
             'Đây là số đo từ các artifact đã hoàn thành. Smoke test không có giá trị chứng minh hiệu quả.', '',
             '## Giao thức', '',
             'Epson, một rig, tốc độ danh định 1200 rpm; XYZ vận tốc mm/s, 3000 Hz. '
             'Cửa sổ 1 giây không chồng lấn, chuẩn hóa bằng scaler train riêng từng k. '
             'Dữ liệu được chia theo bản ghi với seed 42; chọn tập con thiếu dữ liệu với seed 101.', '',
             f"Profiles: {c['profiles']}; seeds: {c['seeds']}. CNN: {c['detector_steps']} cập nhật/nhánh; "
             f"cVAE: {c['generator_steps']} cập nhật; batch {c['batch_size']}; latent {c['latent_dim']}; beta {c['beta']}.", '',
             'A: chỉ dữ liệu thật. B: lấy mẫu lặp. C: nhân biên độ và dịch vòng đồng thời cả XYZ. '
             'D: sinh từ prior N(0,I) của cVAE theo nhãn lỗi. B/C/D có cùng số mẫu thêm mỗi lớp. '
             'Mọi detector có cùng kiến trúc, số cập nhật và seed khởi tạo. A có ít mẫu duy nhất hơn; '
             'cân bằng theo số cập nhật, không theo epoch.', '',
             'Checkpoint chọn bằng validation CE (detector) hoặc MSE + beta × KL (generator). '
             'Score lỗi là tổng xác suất lớp 1–5. Ngưỡng riêng cho cửa sổ và recording, '
             f"chọn trên normal validation với FPR thực nghiệm ≤ {c['max_validation_fpr']:.0%}. "
             'Đây là lựa chọn thí nghiệm, không phải yêu cầu vận hành đã được nhà máy phê duyệt. '
             'Điểm recording là trung bình điểm cửa sổ.', '',
             '## Kết quả theo recording', '',
             f'Tập hiển thị: **{split}**. Mỗi seed đánh giá cùng 24 recording: 4 normal + 20 fault. '
             'Bảng ghi trung bình [min; max] giữa các seed, không phải khoảng tin cậy độc lập.', '',
             '| k | Nhánh | n seed | Recall | FPR | Macro-F1 | AP |',
             '|---|---|---:|---:|---:|---:|---:|']
    def summary(values):
        return f'{np.mean(values):.3f} [{min(values):.3f}; {max(values):.3f}]'
    for profile in c['profiles']:
        for method in METHODS:
            group = [r for r in rows if (r['profile'], r['method'], r['split'], r['level']) == (profile, method, split, 'recording')]
            if group:
                lines.append(f'| {profile} | {DISPLAY[method]} | {len(group)} | '+
                             ' | '.join(summary([r[key] for r in group]) for key in ['recall', 'fpr', 'macro_f1', 'average_precision'])+' |')
    paired = []
    for profile in c['profiles']:
        for seed in c['seeds']:
            matched = {r['method']: r for r in rows if (r['profile'], r['seed'], r['split'], r['level']) == (profile, seed, split, 'recording')}
            if 'cvae' in matched:
                for baseline in METHODS[:-1]:
                    if baseline in matched:
                        paired.append(dict(profile=profile, seed=seed, baseline=baseline,
                                           delta_recall=matched['cvae']['recall']-matched[baseline]['recall'],
                                           delta_fpr=matched['cvae']['fpr']-matched[baseline]['fpr']))
    save_json(out/'paired_differences.json', paired)
    lines.extend(['', '## Diễn giải và giới hạn', '',
                  'Không chọn nhánh tốt nhất dựa trên test. Xem chênh lệch cVAE với từng baseline trong '
                  '`paired_differences.json`; kiểm tra đồng thời recall và FPR. AP là Average Precision '
                  '(tổng có trọng số theo bước recall), không phải tích phân hình thang của đường PR.', '',
                  'Chỉ có 4 normal recording: FPR ở mức recording nhảy theo bước 25%. '
                  'Giới hạn 5% trên validation vì vậy tương đương không báo nhầm recording normal nào. '
                  'Không thể dùng tập nhỏ này để bảo đảm FPR vận hành 5%. Các cửa sổ cùng recording có phụ thuộc.', '',
                  'Ba seed chỉ phản ánh biến thiên khởi tạo/lấy mẫu trên cùng split; không chứng minh '
                  'khả năng tổng quát qua phiên đo, máy khác hay tốc độ khác. Độc lập giữa các phiên ghi chưa xác minh. '
                  'Sáu lớp là một normal và năm cấu hình mất cân bằng, không phải năm cơ chế hỏng độc lập.', '',
                  'Validation vẫn có đủ 20 recording lỗi để chọn mô hình; chỉ phần cập nhật trọng số là few-shot. '
                  'Mẫu sinh không làm tăng số recording độc lập. Không suy ra RUL.', '',
                  '## Kiểm tra dữ liệu sinh', '',
                  '| k | Seed | Số lớp có cảnh báo RMS | Mẫu trùng train chính xác |',
                  '|---|---:|---:|---:|'])
    for q in quality:
        warnings = sum(v['rms_warning'] for v in q['classes'].values())
        duplicates = sum(v['exact_training_duplicates'] for v in q['classes'].values())
        lines.append(f"| {q['profile']} | {q['seed']} | {warnings}/5 | {duplicates} |")
    lines.extend(['', 'Cảnh báo RMS khi tỷ lệ synthetic/real nằm ngoài [0.5; 2] ở ít nhất một kênh; '
                  'chỉ là chẩn đoán đã định trước, không phải chứng nhận vật lý và không dùng để lọc mẫu. '
                  'Đọc thêm PSD, tương quan và độ phân tán trong `synthetic_quality.json`. '
                  'Không có mẫu trùng byte không đồng nghĩa không ghi nhớ tín hiệu.', '',
                  '## Đầu ra và khả năng tái lập', '',
                  '`metrics.csv`: mọi seed, cửa sổ/recording, validation/test hiện có. '
                  '`resources.json`: thời gian và bộ nhớ CUDA từng detector. Checkpoint, đường học, '
                  'ngưỡng, dự đoán đầy đủ và provenance nằm trong từng thư mục stage. '
                  'Lệnh export ghép dữ liệu thật + mẫu thêm thành một NPZ; lệnh demo sinh mới theo seed.', '',
                  '## Nguồn kỹ thuật', '',
                  '- [PyTorch — cài đặt phiên bản trước](https://pytorch.org/get-started/previous-versions/)',
                  '- [VAE — Kingma và Welling](https://arxiv.org/abs/1312.6114)',
                  '- [Conditional VAE — Sohn et al.](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)',
                  '- [scikit-learn — Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)', '',
                  'Thiết kế Conv1D ở đây là baseline triển khai cho dữ liệu này; không tuyên bố tái hiện nguyên bản paper cVAE.'])
    (out/'BAO_CAO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for metric, label in [('recall', 'Fault recall'), ('fpr', 'False-positive rate')]:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        positions, ticks = [], []
        for pidx, profile in enumerate(c['profiles']):
            for midx, method in enumerate(METHODS):
                group = [r for r in rows if (r['profile'], r['method'], r['split'], r['level']) == (profile, method, split, 'recording')]
                x = pidx*5+midx
                positions.append(x)
                ticks.append(profile+'\n'+DISPLAY[method])
                if group:
                    offsets = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else [0]
                    ax.scatter(x+np.array(offsets), [r[metric] for r in group], s=30,
                               color='#0072B2' if method == 'cvae' else '#666666', zorder=3)
        ax.set(xticks=positions, xticklabels=ticks, ylabel=label, ylim=(-0.06, 1.06),
               title=f'Recording level — {split}; each dot = one seed')
        ax.grid(axis='y', color='#dddddd', linewidth=0.5)
        save_figure(fig, out/(metric+'_recording'))
    print(f'REPORT {out / "BAO_CAO.md"}', flush=True)


def demo(args):
    device = device_for(args.device)
    run, c = open_run(args.run)
    if args.profile not in c['profiles'] or args.seed not in c['seeds']:
        raise ValueError('Profile/seed not in this run')
    generator = completed(run, args.profile, args.seed, 'generator')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty demo directory')
    out.mkdir(parents=True, exist_ok=True)
    y = np.array([args.class_id], dtype=np.int64)
    x = generate(generator/'generator.pt', y, c, args.sample_seed, device)
    loader = EpsonPreparedLoader()
    scaler = loader.get_scaler(args.profile)
    raw = x*np.array(scaler['scale'])[None, :, None]+np.array(scaler['mean'])[None, :, None]
    np.savez_compressed(out/'sample.npz', X=x, X_raw_mm_s=raw, y_class=y, sampling_rate_hz=3000)
    time_axis = np.arange(3000)/3000
    np.savetxt(out/'waveform.csv', np.column_stack([time_axis, raw[0].T]), delimiter=',',
               header='time_s,X_mm_s,Y_mm_s,Z_mm_s', comments='')
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for channel, color, name in zip(range(3), ['#0072B2', '#D55E00', '#666666'], ['X', 'Y', 'Z']):
        ax.plot(time_axis, raw[0, channel], label=name, color=color, linewidth=0.65)
    ax.set(xlabel='Time (s)', ylabel='Velocity (mm/s)', title=f'Prior-generated cVAE sample — class {args.class_id}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    save_figure(fig, out/'waveform')
    train = load_verified(loader, args.profile, 'train')
    real = train.X[train.y_class == args.class_id]*np.array(scaler['scale'])[None, :, None]
    real += np.array(scaler['mean'])[None, :, None]
    frequency, psd = welch(real, fs=3000, nperseg=512, axis=-1)
    _, spsd = welch(raw, fs=3000, nperseg=512, axis=-1)
    real_psd, sample_psd = psd.mean((0, 1)), spsd.mean((0, 1))
    np.savetxt(out/'psd.csv', np.column_stack([frequency, real_psd, sample_psd]), delimiter=',',
               header='frequency_hz,real_mean_psd,synthetic_sample_psd', comments='')
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.semilogy(frequency, np.maximum(real_psd, 1e-30), color='#666666', label='Real train mean')
    ax.semilogy(frequency, np.maximum(sample_psd, 1e-30), color='#0072B2', label='Generated sample')
    ax.set(xlabel='Frequency (Hz)', ylabel='Mean XYZ PSD ((mm/s)²/Hz)', title='Welch PSD — descriptive comparison')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    save_figure(fig, out/'psd')
    save_json(out/'demo.json', dict(profile=args.profile, model_seed=args.seed, sample_seed=args.sample_seed,
                                   class_id=args.class_id, source='N(0,I) prior; not a real measurement',
                                   rms_mm_s=np.sqrt((raw**2).mean(-1)).tolist()))
    print(f'DEMO {out}', flush=True)
