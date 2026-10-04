from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import welch, correlate, correlation_lags


# ============================================================
# CONFIGURATION — FIXED BEFORE VIEWING FIGURES
# ============================================================

FS = 3000.0
CHANNELS = ["X", "Y", "Z"]

N_BLOCKS = 10

# Full-waveform display only; raw data are not modified.
MAX_DISPLAY_POINTS = 6000

# Fixed zoom duration = 2 s.
ZOOM_SAMPLES = 6000

# Welch PSD diagnostic configuration.
PSD_NPERSEG = 3000
PSD_NOVERLAP = 1500

# Within-record channel relation diagnostic only.
XCORR_SEGMENT_SAMPLES = 9000      # 3 seconds, middle of record
XCORR_MAX_LAG_SAMPLES = 150       # +/- 0.05 s

# Generic integrity-screen rules.
# These are QA flags, NOT model thresholds and NOT proof of sensor clipping.
NEAR_CONSTANT_STD = 1e-12
CONSTANT_SAMPLE_FRACTION_WARN = 0.01
REPEATED_EXTREME_FRACTION_WARN = 0.005
ROBUST_JUMP_Z = 12.0


PERSON1 = Path(__file__).resolve().parent
PROJECT = PERSON1.parent

PREP_REPORT = PERSON1 / "results" / "prepared_v1" / "preparation_report.json"
MANIFEST = PERSON1 / "results" / "audit_v1" / "public_manifest.csv"

QA_DIR = PERSON1 / "results" / "qa_v2"
FIG_DIR = QA_DIR / "figures_train_k1"

STATS_OUT = QA_DIR / "signal_qc_train_k1.csv"
REL_OUT = QA_DIR / "channel_relations_train_k1.csv"
REVIEW_OUT = QA_DIR / "signal_review.txt"
META_OUT = QA_DIR / "signal_qa_protocol.json"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def constant_run_stats(x):
    """
    Exact-value repeated consecutive samples.

    A run of L identical samples has L-1 consecutive zero differences.
    Returns:
        number of runs with length >= 2
        total samples participating in those runs
        longest run length
    """
    if len(x) < 2:
        return 0, 0, 1

    eq = np.diff(x) == 0
    padded = np.concatenate(([False], eq, [False]))

    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])

    if len(starts) == 0:
        return 0, 0, 1

    # True-run length counts equal transitions.
    # Number of equal samples = transitions + 1.
    lengths = (ends - starts) + 1

    return (
        int(len(lengths)),
        int(np.sum(lengths)),
        int(np.max(lengths)),
    )


def robust_jump_stats(x):
    """
    Screening rule on first differences:

      d = x[t] - x[t-1]
      center = median(d)
      robust_sigma = 1.4826 * median(|d-center|)
      flag if |d-center| > 12 * robust_sigma

    If MAD is zero, standard deviation of d is used only as fallback.
    """
    d = np.diff(x)

    if len(d) == 0:
        return 0, math.nan, math.nan, math.nan

    center = float(np.median(d))
    mad = float(np.median(np.abs(d - center)))
    robust_sigma = 1.4826 * mad

    if robust_sigma <= 0:
        fallback = float(np.std(d))

        if fallback <= 0:
            return 0, math.inf, center, 0.0

        threshold = ROBUST_JUMP_Z * fallback
    else:
        threshold = ROBUST_JUMP_Z * robust_sigma

    count = int(np.sum(np.abs(d - center) > threshold))

    return count, float(threshold), center, float(robust_sigma)


def block_stats(x):
    blocks = np.array_split(x, N_BLOCKS)

    means = np.array([np.mean(b) for b in blocks], dtype=float)
    stds = np.array([np.std(b) for b in blocks], dtype=float)
    rmss = np.array([rms(b) for b in blocks], dtype=float)

    return {
        "block_mean_min": float(np.min(means)),
        "block_mean_max": float(np.max(means)),
        "block_mean_range": float(np.ptp(means)),
        "block_std_min": float(np.min(stds)),
        "block_std_max": float(np.max(stds)),
        "block_rms_min": float(np.min(rmss)),
        "block_rms_max": float(np.max(rmss)),
    }


def safe_record_name(recording_id):
    return recording_id.replace("/", "_").replace("\\", "_")


def plot_record(recording_id, arr):
    n = arr.shape[0]
    t = np.arange(n) / FS

    display_step = max(1, int(math.ceil(n / MAX_DISPLAY_POINTS)))
    display_idx = np.arange(0, n, display_step)

    zoom_len = min(ZOOM_SAMPLES, n)

    zoom_starts = {
        "start": 0,
        "middle": max(0, (n // 2) - (zoom_len // 2)),
        "end": max(0, n - zoom_len),
    }

    fig, axes = plt.subplots(5, 1, figsize=(15, 17))

    for c, name in enumerate(CHANNELS):
        axes[0].plot(t[display_idx], arr[display_idx, c], label=name)

    axes[0].set_title(
        f"{recording_id} | full waveform | display-decimated only"
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Velocity (mm/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for ax_i, segment_name in enumerate(["start", "middle", "end"], start=1):
        s = zoom_starts[segment_name]
        e = min(n, s + zoom_len)
        seg_t = np.arange(s, e) / FS

        for c, name in enumerate(CHANNELS):
            axes[ax_i].plot(seg_t, arr[s:e, c], label=name)

        axes[ax_i].set_title(
            f"{segment_name} zoom | samples [{s}:{e})"
        )
        axes[ax_i].set_xlabel("Time (s)")
        axes[ax_i].set_ylabel("Velocity (mm/s)")
        axes[ax_i].legend()
        axes[ax_i].grid(True, alpha=0.25)

    for c, name in enumerate(CHANNELS):
        f, pxx = welch(
            arr[:, c],
            fs=FS,
            window="hann",
            nperseg=PSD_NPERSEG,
            noverlap=PSD_NOVERLAP,
            detrend="constant",
            scaling="density",
        )

        axes[4].semilogy(f, np.maximum(pxx, np.finfo(float).tiny), label=name)

    axes[4].set_title(
        f"Welch PSD | Fs={FS:g} Hz | Hann | nperseg={PSD_NPERSEG} | overlap={PSD_NOVERLAP}"
    )
    axes[4].set_xlabel("Frequency (Hz)")
    axes[4].set_ylabel("PSD ((mm/s)^2/Hz)")
    axes[4].set_xlim(0, FS / 2)
    axes[4].legend()
    axes[4].grid(True, alpha=0.25)

    fig.tight_layout()

    out = FIG_DIR / f"{safe_record_name(recording_id)}__waveform_psd.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return zoom_starts


def channel_relations(recording_id, arr):
    n = len(arr)
    seg_len = min(XCORR_SEGMENT_SAMPLES, n)

    start = max(0, (n // 2) - (seg_len // 2))
    end = start + seg_len

    seg = arr[start:end].astype(np.float64, copy=False)

    pairs = [(0, 1), (0, 2), (1, 2)]
    rows = []

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    for ax, (a, b) in zip(axes, pairs):
        xa = seg[:, a] - np.mean(seg[:, a])
        xb = seg[:, b] - np.mean(seg[:, b])

        denom = np.linalg.norm(xa) * np.linalg.norm(xb)

        if denom == 0:
            norm_corr = np.full(2 * len(xa) - 1, np.nan)
        else:
            norm_corr = correlate(xa, xb, mode="full", method="fft") / denom

        lags = correlation_lags(len(xa), len(xb), mode="full")

        mask = np.abs(lags) <= XCORR_MAX_LAG_SAMPLES
        local_lags = lags[mask]
        local_corr = norm_corr[mask]

        finite = np.isfinite(local_corr)

        if finite.any():
            idx_local = np.nanargmax(np.abs(local_corr))
            best_corr = float(local_corr[idx_local])
            best_lag = int(local_lags[idx_local])
        else:
            best_corr = math.nan
            best_lag = 0

        zero_corr = np.corrcoef(seg[:, a], seg[:, b])[0, 1]
        zero_corr = float(zero_corr) if np.isfinite(zero_corr) else math.nan

        rows.append({
            "recording_id": recording_id,
            "segment_start": start,
            "segment_end_exclusive": end,
            "channel_a": CHANNELS[a],
            "channel_b": CHANNELS[b],
            "pearson_zero_lag": zero_corr,
            "max_abs_xcorr_value_signed": best_corr,
            "max_abs_xcorr_lag_samples": best_lag,
            "max_abs_xcorr_lag_seconds": best_lag / FS,
            "max_lag_screen_samples": XCORR_MAX_LAG_SAMPLES,
        })

        ax.plot(local_lags / FS, local_corr)
        ax.axvline(0.0, linewidth=0.8)
        ax.set_title(
            f"{CHANNELS[a]} vs {CHANNELS[b]} | middle samples [{start}:{end})"
        )
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized cross-correlation")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{recording_id} | within-record channel relation diagnostic\n"
        "No channel is shifted or modified in stored data",
        y=1.01,
    )

    fig.tight_layout()

    out = FIG_DIR / f"{safe_record_name(recording_id)}__channel_relation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return rows


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with PREP_REPORT.open("r", encoding="utf-8") as f:
        prep = json.load(f)

    manifest = pd.read_csv(MANIFEST)

    train_ids = list(prep["budgets"]["1"]["train_ids"])

    if len(train_ids) != 13:
        raise RuntimeError(
            f"Expected 13 k1 train recordings, found {len(train_ids)}"
        )

    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Duplicate recording ID inside k1 train list")

    selected = manifest[manifest["recording_id"].isin(train_ids)].copy()

    if len(selected) != len(train_ids):
        missing = sorted(set(train_ids) - set(selected["recording_id"]))
        raise RuntimeError(f"Manifest missing k1 IDs: {missing}")

    class_counts = (
        selected.groupby("class_id")["recording_id"]
        .count()
        .sort_index()
        .to_dict()
    )

    expected_counts = {0: 8, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    if class_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected k1 class counts: {class_counts}; expected {expected_counts}"
        )

    stats_rows = []
    relation_rows = []
    auto_flags = []

    plot_metadata = {}

    for record_index, recording_id in enumerate(train_ids, start=1):
        row = selected[selected["recording_id"] == recording_id].iloc[0]

        path = Path(row["filepath"])

        if not path.exists():
            raise FileNotFoundError(path)

        print(
            f"[{record_index:02d}/{len(train_ids)}] "
            f"Reading {recording_id}"
        )

        arr = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
        )

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise RuntimeError(
                f"{recording_id}: expected [N,3], got {arr.shape}"
            )

        n = arr.shape[0]

        manifest_n = int(row["n_samples"])

        if n != manifest_n:
            raise RuntimeError(
                f"{recording_id}: CSV samples={n}, manifest={manifest_n}"
            )

        if not np.isfinite(arr).all():
            auto_flags.append(
                f"{recording_id}: NONFINITE values found"
            )

        zoom_starts = plot_record(recording_id, arr)
        plot_metadata[recording_id] = {
            "n_samples": int(n),
            "full_display_max_points": MAX_DISPLAY_POINTS,
            "zoom_length_samples": min(ZOOM_SAMPLES, n),
            "zoom_starts": zoom_starts,
        }

        relation_rows.extend(
            channel_relations(recording_id, arr)
        )

        for c, channel in enumerate(CHANNELS):
            x = arr[:, c]

            finite_count = int(np.isfinite(x).sum())
            nonfinite_count = int(len(x) - finite_count)

            x_min = float(np.min(x))
            x_max = float(np.max(x))
            x_std = float(np.std(x))

            const_runs, const_samples, longest_const = constant_run_stats(x)
            const_fraction = const_samples / len(x)

            extreme_count = int(
                np.sum((x == x_min) | (x == x_max))
            )
            extreme_fraction = extreme_count / len(x)

            (
                jump_count,
                jump_threshold,
                diff_center,
                diff_robust_sigma,
            ) = robust_jump_stats(x)

            bstats = block_stats(x)

            flags = []

            if nonfinite_count > 0:
                flags.append("NONFINITE")

            if x_std <= NEAR_CONSTANT_STD:
                flags.append("NEAR_CONSTANT")

            if const_fraction >= CONSTANT_SAMPLE_FRACTION_WARN:
                flags.append("MANY_EXACT_CONSTANT_SAMPLES")

            if extreme_fraction >= REPEATED_EXTREME_FRACTION_WARN:
                flags.append("REPEATED_EXACT_EXTREMES_SCREEN")

            if flags:
                auto_flags.append(
                    f"{recording_id} / {channel}: " + ";".join(flags)
                )

            stats_rows.append({
                "recording_id": recording_id,
                "class_id": int(row["class_id"]),
                "config_label": row["config_label"],
                "channel": channel,
                "unit": "mm/s",
                "fs_hz": FS,
                "n_samples": int(n),
                "duration_s": n / FS,

                "mean": float(np.mean(x)),
                "std": x_std,
                "rms": rms(x),
                "peak_to_peak": float(np.ptp(x)),
                "min": x_min,
                "max": x_max,

                "nonfinite_count": nonfinite_count,

                "constant_run_count": const_runs,
                "constant_run_total_samples": const_samples,
                "constant_sample_fraction": const_fraction,
                "longest_constant_run_samples": longest_const,

                "exact_extreme_count": extreme_count,
                "exact_extreme_fraction": extreme_fraction,

                "large_jump_count": jump_count,
                "jump_threshold_abs_mm_s_per_sample": jump_threshold,
                "diff_median": diff_center,
                "diff_robust_sigma": diff_robust_sigma,

                **bstats,

                "automatic_flags": ";".join(flags),
            })

    stats_df = pd.DataFrame(stats_rows)
    rel_df = pd.DataFrame(relation_rows)

    stats_df.to_csv(STATS_OUT, index=False)
    rel_df.to_csv(REL_OUT, index=False)

    protocol = {
        "scope": "k1 train only",
        "train_record_count": len(train_ids),
        "train_ids": train_ids,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "fs_hz": FS,
        "channels": CHANNELS,
        "raw_unit": "mm/s",
        "rules_fixed_before_figures": True,

        "time_block_count": N_BLOCKS,

        "waveform_display": {
            "max_display_points": MAX_DISPLAY_POINTS,
            "note": "downsampling is for plotting only; raw/training data unchanged",
        },

        "zoom_samples": ZOOM_SAMPLES,

        "welch_psd": {
            "window": "hann",
            "nperseg": PSD_NPERSEG,
            "noverlap": PSD_NOVERLAP,
            "scaling": "density",
        },

        "jump_screen": {
            "formula": (
                "|diff - median(diff)| > "
                f"{ROBUST_JUMP_Z} * robust_sigma; "
                "robust_sigma=1.4826*MAD(diff)"
            ),
            "purpose": "integrity screening only",
        },

        "constant_screen": {
            "exact_value_comparison": True,
            "warning_fraction": CONSTANT_SAMPLE_FRACTION_WARN,
        },

        "extreme_screen": {
            "warning_fraction": REPEATED_EXTREME_FRACTION_WARN,
            "interpretation": (
                "screening flag only; does not prove physical sensor clipping"
            ),
        },

        "channel_relation": {
            "segment_selection": "fixed middle segment",
            "segment_samples": XCORR_SEGMENT_SAMPLES,
            "max_lag_samples": XCORR_MAX_LAG_SAMPLES,
            "note": (
                "diagnostic only; channels are not independently shifted "
                "or realigned in stored data"
            ),
        },

        "figures": plot_metadata,

        "prohibited_usage": [
            "No validation/test figures used for preprocessing design",
            "No automatic sample deletion",
            "No raw data modification",
            "No conclusion of physical clipping without sensor range evidence",
        ],
    }

    with META_OUT.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    flagged_rows = stats_df[
        stats_df["automatic_flags"].fillna("").astype(str).str.len() > 0
    ]

    lines = []

    lines.append("EPSON SIGNAL QA — K1 TRAIN REVIEW")
    lines.append("=" * 72)
    lines.append("")
    lines.append("SCOPE")
    lines.append("Only the 13 real train recordings allowed in k1 were inspected.")
    lines.append("Validation/test waveforms were NOT used.")
    lines.append("")
    lines.append(
        "Class counts: "
        + ", ".join(
            f"class {k}={v}" for k, v in sorted(class_counts.items())
        )
    )
    lines.append("")

    lines.append("AUTOMATED CHECKS")
    lines.append("- Mean/std/RMS/peak-to-peak/min/max per channel.")
    lines.append(f"- {N_BLOCKS} fixed time blocks per channel.")
    lines.append("- Exact consecutive constant-value runs.")
    lines.append("- Exact repeated min/max-value fraction.")
    lines.append(
        "- First-difference robust jump screen using median/MAD; "
        f"multiplier={ROBUST_JUMP_Z:g}."
    )
    lines.append(
        f"- Welch PSD: Fs={FS:g}, Hann, "
        f"nperseg={PSD_NPERSEG}, overlap={PSD_NOVERLAP}."
    )
    lines.append(
        "- Fixed start/middle/end waveform zooms; full waveform "
        "is display-decimated only."
    )
    lines.append(
        "- Within-record channel Pearson/cross-correlation diagnostic "
        "on a fixed middle segment."
    )
    lines.append("")

    lines.append("AUTOMATIC FLAG SUMMARY")
    if len(flagged_rows) == 0:
        lines.append(
            "No automatic integrity flags were triggered under the fixed "
            "screening rules."
        )
    else:
        for _, r in flagged_rows.iterrows():
            lines.append(
                f"- {r['recording_id']} / {r['channel']}: "
                f"{r['automatic_flags']}"
            )

    lines.append("")
    lines.append("INTERPRETATION BOUNDARY")
    lines.append(
        "An automatic flag is a request for review, not evidence that a "
        "sample must be deleted."
    )
    lines.append(
        "Repeated extrema can suggest a clipping-like pattern, but physical "
        "sensor clipping cannot be confirmed without suitable sensor-range/"
        "configuration evidence."
    )
    lines.append(
        "No timestamp exists in the CSV, so timestamp jitter or packet-loss "
        "timing cannot be measured from these files."
    )
    lines.append(
        "This QA does not establish recording-session independence and does "
        "not replace the near/shifted duplicate analysis in Chặng 3."
    )
    lines.append("")
    lines.append("MANUAL REVIEW REQUIRED")
    lines.append(
        "Review waveform/PSD/channel-relation figures for each of the 13 "
        "k1 train recordings."
    )
    lines.append(
        "For any suspicious finding, classify it as: verified data error, "
        "valid signal characteristic, or unresolved."
    )
    lines.append(
        "Do not delete/interpolate/clip samples silently. Any preprocessing "
        "change requires a new prepared-data version."
    )

    REVIEW_OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("============================================================")
    print("QA_COMPLETE")
    print("============================================================")
    print(f"k1 recordings       : {len(train_ids)}")
    print(f"signal QC rows      : {len(stats_df)}")
    print(f"channel relation rows: {len(rel_df)}")
    print(f"automatic flagged rows: {len(flagged_rows)}")
    print(f"figure files        : {len(list(FIG_DIR.glob('*.png')))}")
    print("")
    print(f"STATS   : {STATS_OUT}")
    print(f"REL     : {REL_OUT}")
    print(f"REVIEW  : {REVIEW_OUT}")
    print(f"PROTOCOL: {META_OUT}")
    print(f"FIGURES : {FIG_DIR}")


if __name__ == "__main__":
    main()
