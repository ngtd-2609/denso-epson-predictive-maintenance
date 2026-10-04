from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch

# ============================================================
# SETTINGS FIXED BEFORE REAL-DATA SCREEN
# ============================================================

FS = 3000.0

N_SEGMENTS = 5
SEGMENT_LENGTH = 6000

PSD_NPERSEG = 3000
PSD_NOVERLAP = 1500

PSD_BANDS = [
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 1500.0),
]

TOP_K_PER_RECORD = 5
TOP_GLOBAL = 40
TOP_CROSS_SPLIT = 80

CHANNELS = ["X", "Y", "Z"]

PERSON1 = Path(__file__).resolve().parent

MANIFEST_PATH = (
    PERSON1 / "results" / "audit_v1" / "public_manifest.csv"
)

SPLIT_PATH = (
    PERSON1 / "results" / "split_seed42" / "split.csv"
)

QA_DIR = PERSON1 / "results" / "qa_v2"

FEATURES_OUT = QA_DIR / "near_duplicate_features.csv"
ALL_PAIRS_OUT = QA_DIR / "near_duplicate_all_pair_distances.csv"
CANDIDATES_OUT = QA_DIR / "near_duplicate_candidates.csv"
SUMMARY_OUT = QA_DIR / "near_duplicate_screening_summary.json"
REVIEW_OUT = QA_DIR / "near_duplicate_review.txt"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def safe_corr(a, b):
    sa = np.std(a)
    sb = np.std(b)

    if sa <= 0 or sb <= 0:
        return 0.0

    r = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(r):
        return 0.0

    return float(r)


def band_power(f, pxx, lo, hi):
    mask = (f >= lo) & (f < hi)

    if np.sum(mask) < 2:
        return 0.0

    return float(np.trapezoid(pxx[mask], f[mask]))


def segment_features(seg, segment_index):
    """
    Features are used only for candidate screening.

    Labels and split membership are NOT used to construct the signal
    features or distances.
    """
    out = {}

    for c, ch in enumerate(CHANNELS):
        x = seg[:, c]

        prefix = f"s{segment_index}_{ch}"

        out[f"{prefix}_mean"] = float(np.mean(x))
        out[f"{prefix}_std"] = float(np.std(x))
        out[f"{prefix}_rms"] = rms(x)
        out[f"{prefix}_p2p"] = float(np.ptp(x))

        f, pxx = welch(
            x,
            fs=FS,
            window="hann",
            nperseg=PSD_NPERSEG,
            noverlap=PSD_NOVERLAP,
            detrend="constant",
            scaling="density",
        )

        for band_index, (lo, hi) in enumerate(PSD_BANDS):
            out[
                f"{prefix}_band{band_index}_{int(lo)}_{int(hi)}"
            ] = band_power(f, pxx, lo, hi)

        dom_mask = (f >= 1.0) & (f <= 300.0)

        if np.any(dom_mask):
            local_f = f[dom_mask]
            local_p = pxx[dom_mask]

            dom_i = int(np.argmax(local_p))

            out[f"{prefix}_dominant_freq_hz"] = float(local_f[dom_i])
            out[f"{prefix}_dominant_psd"] = float(local_p[dom_i])
        else:
            out[f"{prefix}_dominant_freq_hz"] = 0.0
            out[f"{prefix}_dominant_psd"] = 0.0

    out[f"s{segment_index}_corr_XY"] = safe_corr(
        seg[:, 0],
        seg[:, 1],
    )

    out[f"s{segment_index}_corr_XZ"] = safe_corr(
        seg[:, 0],
        seg[:, 2],
    )

    out[f"s{segment_index}_corr_YZ"] = safe_corr(
        seg[:, 1],
        seg[:, 2],
    )

    return out


def get_fixed_starts(n_samples):
    if n_samples < SEGMENT_LENGTH:
        raise RuntimeError(
            f"Recording only has {n_samples} samples, "
            f"shorter than {SEGMENT_LENGTH}"
        )

    max_start = n_samples - SEGMENT_LENGTH

    starts = np.linspace(
        0,
        max_start,
        N_SEGMENTS,
        dtype=np.int64,
    )

    return starts.tolist()


def robust_feature_scale(X):
    """
    Robust scaling across the 96 recordings for integrity screening only.

    This is NOT the detector scaler and is NOT written into prepared data.
    """
    center = np.median(X, axis=0)

    mad = np.median(
        np.abs(X - center[None, :]),
        axis=0,
    )

    scale = 1.4826 * mad

    fallback_std = np.std(X, axis=0)

    bad = (~np.isfinite(scale)) | (scale < 1e-15)

    scale[bad] = fallback_std[bad]

    bad = (~np.isfinite(scale)) | (scale < 1e-15)

    scale[bad] = 1.0

    return center, scale


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    split = pd.read_csv(SPLIT_PATH)

    if len(manifest) != 96:
        raise RuntimeError(
            f"Expected 96 manifest records, found {len(manifest)}"
        )

    if len(split) != 96:
        raise RuntimeError(
            f"Expected 96 split records, found {len(split)}"
        )

    if manifest["recording_id"].duplicated().any():
        raise RuntimeError("Duplicate recording_id in manifest")

    if split["recording_id"].duplicated().any():
        raise RuntimeError("Duplicate recording_id in split")

    split_map = split.set_index("recording_id")["split"].to_dict()

    missing_split = sorted(
        set(manifest["recording_id"])
        - set(split_map)
    )

    if missing_split:
        raise RuntimeError(
            f"Missing split assignments: {missing_split}"
        )

    feature_rows = []

    print("=" * 72)
    print("EPSON NEAR-DUPLICATE CANDIDATE SCREEN")
    print("=" * 72)
    print("Recordings:", len(manifest))
    print("Segments/record:", N_SEGMENTS)
    print("Segment length:", SEGMENT_LENGTH)
    print("No label or model performance is used in feature distances.")
    print()

    for idx, row in manifest.iterrows():
        recording_id = str(row["recording_id"])
        path = Path(row["filepath"])

        print(
            f"[{idx+1:02d}/96] {recording_id}"
        )

        arr = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
        )

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise RuntimeError(
                f"{recording_id}: invalid shape {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{recording_id}: non-finite values"
            )

        starts = get_fixed_starts(len(arr))

        features = {
            "recording_id": recording_id,
            "n_samples": int(len(arr)),
        }

        for segment_index, start in enumerate(starts):
            end = start + SEGMENT_LENGTH

            seg = arr[start:end]

            sf = segment_features(
                seg,
                segment_index,
            )

            features.update(sf)

            features[
                f"s{segment_index}_start_sample"
            ] = int(start)

            features[
                f"s{segment_index}_end_exclusive"
            ] = int(end)

        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)

    metadata_cols = (
        ["recording_id", "n_samples"]
        + [
            f"s{i}_start_sample"
            for i in range(N_SEGMENTS)
        ]
        + [
            f"s{i}_end_exclusive"
            for i in range(N_SEGMENTS)
        ]
    )

    feature_cols = [
        c for c in feat_df.columns
        if c not in metadata_cols
    ]

    X = feat_df[feature_cols].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite feature matrix")

    center, scale = robust_feature_scale(X)

    Z = (X - center[None, :]) / scale[None, :]

    ids = feat_df["recording_id"].tolist()

    pair_rows = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            diff = Z[i] - Z[j]

            # RMS standardized feature distance.
            distance = float(
                np.sqrt(np.mean(diff * diff))
            )

            a = ids[i]
            b = ids[j]

            split_a = split_map[a]
            split_b = split_map[b]

            pair_rows.append({
                "recording_a": a,
                "recording_b": b,
                "split_a": split_a,
                "split_b": split_b,
                "cross_split": split_a != split_b,
                "feature_distance": distance,
            })

    pairs = pd.DataFrame(pair_rows)

    pairs = pairs.sort_values(
        [
            "feature_distance",
            "recording_a",
            "recording_b",
        ]
    ).reset_index(drop=True)

    pairs["global_rank"] = np.arange(
        1,
        len(pairs) + 1,
    )

    manifest_annotation = manifest.set_index(
        "recording_id"
    )[
        ["class_id", "config_label"]
    ].to_dict("index")

    pairs["class_id_a"] = pairs["recording_a"].map(
        lambda x: manifest_annotation[x]["class_id"]
    )

    pairs["class_id_b"] = pairs["recording_b"].map(
        lambda x: manifest_annotation[x]["class_id"]
    )

    pairs["config_a"] = pairs["recording_a"].map(
        lambda x: manifest_annotation[x]["config_label"]
    )

    pairs["config_b"] = pairs["recording_b"].map(
        lambda x: manifest_annotation[x]["config_label"]
    )

    selected_reasons = {}

    def add_reason(index, reason):
        selected_reasons.setdefault(index, set()).add(reason)

    # Fixed top-K coverage for every recording.
    for recording_id in ids:
        local = pairs[
            (pairs["recording_a"] == recording_id)
            |
            (pairs["recording_b"] == recording_id)
        ].nsmallest(
            TOP_K_PER_RECORD,
            "feature_distance",
        )

        for pair_index in local.index:
            add_reason(
                pair_index,
                "TOP5_FOR_RECORD",
            )

    # Fixed global nearest pairs.
    for pair_index in pairs.nsmallest(
        TOP_GLOBAL,
        "feature_distance",
    ).index:
        add_reason(
            pair_index,
            "TOP_GLOBAL",
        )

    # Leakage-relevant cross-split candidates receive
    # explicit coverage.
    cross = pairs[
        pairs["cross_split"] == True
    ].nsmallest(
        TOP_CROSS_SPLIT,
        "feature_distance",
    )

    for pair_index in cross.index:
        add_reason(
            pair_index,
            "TOP_CROSS_SPLIT",
        )

    candidate_indices = sorted(selected_reasons)

    candidates = pairs.loc[
        candidate_indices
    ].copy()

    candidates["selection_reason"] = [
        ";".join(
            sorted(selected_reasons[i])
        )
        for i in candidate_indices
    ]

    candidates["screen_status"] = "SCREEN_CANDIDATE"
    candidates["verification_status"] = "NOT_YET_VERIFIED"

    candidates = candidates.sort_values(
        [
            "feature_distance",
            "recording_a",
            "recording_b",
        ]
    ).reset_index(drop=True)

    feat_df.to_csv(
        FEATURES_OUT,
        index=False,
    )

    pairs.to_csv(
        ALL_PAIRS_OUT,
        index=False,
    )

    candidates.to_csv(
        CANDIDATES_OUT,
        index=False,
    )

    summary = {
        "status": "COMPLETE",
        "scope": "all 96 Epson trimmed recordings",
        "record_count": len(ids),
        "all_pair_count": len(pairs),
        "candidate_count": len(candidates),
        "cross_split_candidate_count": int(
            candidates["cross_split"].sum()
        ),

        "settings_fixed_before_scan": {
            "fs_hz": FS,
            "segment_count": N_SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "psd_nperseg": PSD_NPERSEG,
            "psd_noverlap": PSD_NOVERLAP,
            "psd_bands": PSD_BANDS,
            "top_k_per_record": TOP_K_PER_RECORD,
            "top_global": TOP_GLOBAL,
            "top_cross_split": TOP_CROSS_SPLIT,
        },

        "feature_count": len(feature_cols),

        "distance": (
            "RMS Euclidean distance after robust per-feature scaling "
            "across recordings; screening only"
        ),

        "important_notes": [
            "Labels and split membership were not used to construct signal features.",
            "Class/split fields are annotations for review only.",
            "Feature proximity is not proof of duplicate provenance.",
            "No recording was deleted or reassigned.",
            "Real-pair verification must use one common lag for all three channels.",
        ],
    }

    with SUMMARY_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = []

    lines.append(
        "EPSON NEAR-DUPLICATE CANDIDATE SCREEN"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"Records scanned: {len(ids)}"
    )
    lines.append(
        f"All unique pairs: {len(pairs)}"
    )
    lines.append(
        f"Screen candidates: {len(candidates)}"
    )
    lines.append(
        "Cross-split candidates: "
        f"{int(candidates['cross_split'].sum())}"
    )
    lines.append("")
    lines.append(
        "Candidate status does NOT mean duplicate."
    )
    lines.append(
        "Candidates require long-segment common-lag verification."
    )
    lines.append("")
    lines.append(
        "Top 20 cross-split candidates by feature distance:"
    )

    top_cross = candidates[
        candidates["cross_split"] == True
    ].nsmallest(
        20,
        "feature_distance",
    )

    for _, r in top_cross.iterrows():
        lines.append(
            f"- {r['recording_a']} [{r['split_a']}] "
            f"<-> {r['recording_b']} [{r['split_b']}], "
            f"distance={r['feature_distance']:.6f}, "
            f"classes={r['class_id_a']}/{r['class_id_b']}"
        )

    lines.append("")
    lines.append(
        "No source/session relationship is concluded from this screen."
    )

    REVIEW_OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SCREEN COMPLETE")
    print("=" * 72)
    print("Feature count         :", len(feature_cols))
    print("All unique pairs      :", len(pairs))
    print("Candidates            :", len(candidates))
    print(
        "Cross-split candidates:",
        int(candidates["cross_split"].sum()),
    )

    print()
    print("TOP 20 CROSS-SPLIT CANDIDATES")
    print()

    print(
        top_cross[
            [
                "recording_a",
                "recording_b",
                "split_a",
                "split_b",
                "class_id_a",
                "class_id_b",
                "feature_distance",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(FEATURES_OUT)
    print(ALL_PAIRS_OUT)
    print(CANDIDATES_OUT)
    print(SUMMARY_OUT)
    print(REVIEW_OUT)


if __name__ == "__main__":
    main()
