from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

FS = 3000.0
SEGMENT_LENGTH = 30000
MAX_LAG = 6000
TOP_N = 20
NOMINAL_PERIOD = FS / 20.0

ROOT = Path(r"D:\DENSO_FACTORY\person1_data")
QA = ROOT / "results" / "qa_v2"

MANIFEST = ROOT / "results" / "audit_v1" / "public_manifest.csv"
VERIFY = QA / "near_duplicate_verification.csv"

DETAIL_OUT = QA / "near_duplicate_multisegment_top20_detail.csv"
SUMMARY_OUT = QA / "near_duplicate_multisegment_top20_summary.csv"
JSON_OUT = QA / "near_duplicate_multisegment_top20_summary.json"


def zscore(x):
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)

    if np.any(sd <= 0):
        raise RuntimeError("Zero-variance channel")

    return (x - mu) / sd


def estimate_common_lag(a, b):
    za = zscore(a)
    zb = zscore(b)

    joint = None
    lags = None

    for c in range(3):
        cc = correlate(
            zb[:, c],
            za[:, c],
            mode="full",
            method="fft"
        )

        lg = correlation_lags(
            len(zb[:, c]),
            len(za[:, c]),
            mode="full"
        )

        denom = np.linalg.norm(za[:, c]) * np.linalg.norm(zb[:, c])

        if denom <= 0:
            raise RuntimeError("Invalid correlation denominator")

        cc = cc / denom

        if joint is None:
            joint = cc
            lags = lg
        else:
            joint += cc

    joint /= 3.0

    mask = np.abs(lags) <= MAX_LAG
    local_lags = lags[mask]
    local_joint = joint[mask]

    i = int(np.nanargmax(local_joint))

    return int(local_lags[i]), float(local_joint[i])


def align(a, b, lag):
    if lag == 0:
        return a, b

    if lag > 0:
        return a[:-lag], b[lag:]

    k = -lag
    return a[k:], b[:-k]


def verify_segment(a, b):
    lag, lag_score = estimate_common_lag(a, b)

    aa, bb = align(a, b, lag)

    za = zscore(aa)
    zb = zscore(bb)

    corr = [
        float(np.corrcoef(za[:, c], zb[:, c])[0, 1])
        for c in range(3)
    ]

    residual = za - zb

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    return {
        "common_lag_samples": lag,
        "common_lag_seconds": lag / FS,
        "lag_score": lag_score,
        "corr_X": corr[0],
        "corr_Y": corr[1],
        "corr_Z": corr[2],
        "mean_channel_corr": float(np.mean(corr)),
        "min_channel_corr": float(np.min(corr)),
        "joint_standardized_rmse": rmse,
        "joint_standardized_mae": mae,
        "matched_samples": int(len(aa)),
    }


def fixed_segments(arr):
    n = len(arr)

    if n < SEGMENT_LENGTH:
        raise RuntimeError(f"Recording too short: {n}")

    starts = {
        "start": 0,
        "middle": (n - SEGMENT_LENGTH) // 2,
        "end": n - SEGMENT_LENGTH,
    }

    result = {}

    for name, s in starts.items():
        result[name] = arr[s:s + SEGMENT_LENGTH].copy()

    return result, starts


def main():
    manifest = pd.read_csv(MANIFEST)
    verification = pd.read_csv(VERIFY)

    cross_mask = (
        verification["cross_split"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    cross = (
        verification[cross_mask]
        .sort_values("review_rank")
        .head(TOP_N)
        .copy()
    )

    if len(cross) != TOP_N:
        raise RuntimeError(
            f"Expected {TOP_N} cross-split pairs, found {len(cross)}"
        )

    manifest_map = (
        manifest
        .set_index("recording_id")
        .to_dict("index")
    )

    needed = sorted(
        set(cross["recording_a"])
        |
        set(cross["recording_b"])
    )

    print("=" * 78)
    print("MULTI-SEGMENT REVIEW - TOP 20 CROSS-SPLIT")
    print("=" * 78)
    print("Pairs:", len(cross))
    print("Unique recordings:", len(needed))
    print("Segments: START / MIDDLE / END")
    print("Segment length:", SEGMENT_LENGTH, "samples = 10 s")
    print("Max common lag:", MAX_LAG, "samples = 2 s")
    print("Nominal 20 Hz period:", NOMINAL_PERIOD, "samples")
    print("No independent X/Y/Z lag.")
    print("No automatic duplicate decision.")
    print()

    cache = {}
    starts_meta = {}

    for i, rid in enumerate(needed, start=1):
        path = Path(manifest_map[rid]["filepath"])

        print(f"[LOAD {i:02d}/{len(needed)}] {rid}")

        arr = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64
        )

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise RuntimeError(f"{rid}: invalid shape {arr.shape}")

        if not np.isfinite(arr).all():
            raise RuntimeError(f"{rid}: non-finite samples")

        segs, starts = fixed_segments(arr)

        cache[rid] = segs
        starts_meta[rid] = starts

    detail_rows = []

    print()
    print("========== VERIFY PAIRS ==========")

    for num, (_, pair) in enumerate(cross.iterrows(), start=1):
        a = pair["recording_a"]
        b = pair["recording_b"]

        print(f"[PAIR {num:02d}/20] {a} <-> {b}")

        for seg_name in ["start", "middle", "end"]:
            metrics = verify_segment(
                cache[a][seg_name],
                cache[b][seg_name]
            )

            detail_rows.append({
                "original_review_rank": int(pair["review_rank"]),
                "recording_a": a,
                "recording_b": b,
                "split_a": pair["split_a"],
                "split_b": pair["split_b"],
                "class_id_a": int(pair["class_id_a"]),
                "class_id_b": int(pair["class_id_b"]),
                "segment": seg_name,
                "a_start_sample": starts_meta[a][seg_name],
                "b_start_sample": starts_meta[b][seg_name],
                **metrics
            })

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(DETAIL_OUT, index=False)

    summary_rows = []

    keys = [
        "original_review_rank",
        "recording_a",
        "recording_b",
        "split_a",
        "split_b",
        "class_id_a",
        "class_id_b",
    ]

    for group_key, g in detail.groupby(keys, sort=False):
        info = dict(zip(keys, group_key))
        g = g.set_index("segment")

        lags = [
            int(g.loc[s, "common_lag_samples"])
            for s in ["start", "middle", "end"]
        ]

        corrs = [
            float(g.loc[s, "mean_channel_corr"])
            for s in ["start", "middle", "end"]
        ]

        rmses = [
            float(g.loc[s, "joint_standardized_rmse"])
            for s in ["start", "middle", "end"]
        ]

        info.update({
            "lag_start": lags[0],
            "lag_middle": lags[1],
            "lag_end": lags[2],

            "lag_range_samples": max(lags) - min(lags),
            "lag_range_revolutions_nominal":
                (max(lags) - min(lags)) / NOMINAL_PERIOD,

            "corr_start": corrs[0],
            "corr_middle": corrs[1],
            "corr_end": corrs[2],

            "corr_mean_3seg": float(np.mean(corrs)),
            "corr_min_3seg": float(np.min(corrs)),
            "corr_max_3seg": float(np.max(corrs)),
            "corr_range_3seg": float(np.ptp(corrs)),

            "rmse_start": rmses[0],
            "rmse_middle": rmses[1],
            "rmse_end": rmses[2],

            "rmse_mean_3seg": float(np.mean(rmses)),
            "rmse_max_3seg": float(np.max(rmses)),
        })

        summary_rows.append(info)

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        ["corr_mean_3seg", "rmse_mean_3seg"],
        ascending=[False, True]
    ).reset_index(drop=True)

    summary["multisegment_review_rank"] = np.arange(
        1,
        len(summary) + 1
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    meta = {
        "status": "MULTISEGMENT_MEASUREMENT_COMPLETE_REVIEW_REQUIRED",
        "pair_count": int(len(summary)),
        "segments": ["start", "middle", "end"],
        "segment_length_samples": SEGMENT_LENGTH,
        "max_common_lag_samples": MAX_LAG,
        "common_lag_per_segment_for_all_xyz": True,
        "automatic_duplicate_decision": None,
        "automatic_delete": False,
        "automatic_split_change": False,
    }

    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    cols = [
        "multisegment_review_rank",
        "original_review_rank",
        "recording_a",
        "recording_b",
        "lag_start",
        "lag_middle",
        "lag_end",
        "lag_range_samples",
        "corr_start",
        "corr_middle",
        "corr_end",
        "corr_mean_3seg",
        "corr_min_3seg",
        "rmse_mean_3seg",
    ]

    print()
    print("=" * 78)
    print("MULTI-SEGMENT MEASUREMENT COMPLETE")
    print("=" * 78)

    print(
        summary[cols].to_string(
            index=False
        )
    )

    print()
    print("Saved:", DETAIL_OUT)
    print("Saved:", SUMMARY_OUT)
    print("Saved:", JSON_OUT)


if __name__ == "__main__":
    main()
