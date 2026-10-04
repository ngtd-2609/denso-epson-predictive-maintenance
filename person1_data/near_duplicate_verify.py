from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.signal import correlate, correlation_lags


# ============================================================
# REAL-PAIR VERIFICATION
# Fixed now, before viewing verification scores.
# ============================================================

FS = 3000.0

SEGMENT_LENGTH = 30000       # 10 s
MAX_LAG = 6000               # +/- 2 s

PERSON1 = Path(__file__).resolve().parent

MANIFEST_PATH = (
    PERSON1 / "results" / "audit_v1" / "public_manifest.csv"
)

CANDIDATES_PATH = (
    PERSON1 / "results" / "qa_v2" / "near_duplicate_candidates.csv"
)

QA_DIR = PERSON1 / "results" / "qa_v2"

VERIFY_OUT = (
    QA_DIR / "near_duplicate_verification.csv"
)

SUMMARY_OUT = (
    QA_DIR / "near_duplicate_verification_summary.json"
)

REVIEW_PATH = (
    QA_DIR / "near_duplicate_review.txt"
)


def zscore_channels(x):
    mu = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    sd = np.std(
        x,
        axis=0,
        keepdims=True,
    )

    if np.any(sd <= 0):
        raise RuntimeError(
            "Zero-variance channel during verification"
        )

    return (x - mu) / sd


def get_middle_segment(arr):
    n = len(arr)

    if n < SEGMENT_LENGTH:
        raise RuntimeError(
            f"Recording has {n} samples, shorter than "
            f"{SEGMENT_LENGTH}"
        )

    start = (n - SEGMENT_LENGTH) // 2
    end = start + SEGMENT_LENGTH

    return (
        arr[start:end].astype(
            np.float64,
            copy=False,
        ),
        int(start),
        int(end),
    )


def estimate_common_lag(a, b):
    """
    Estimate ONE lag jointly across X/Y/Z.

    Positive lag means b is delayed relative to a under
    this implementation convention.

    No independent per-channel lag is permitted.
    """

    za = zscore_channels(a)
    zb = zscore_channels(b)

    joint = None
    lags = None

    for c in range(3):

        cc = correlate(
            zb[:, c],
            za[:, c],
            mode="full",
            method="fft",
        )

        lg = correlation_lags(
            len(zb[:, c]),
            len(za[:, c]),
            mode="full",
        )

        denom = (
            np.linalg.norm(za[:, c])
            *
            np.linalg.norm(zb[:, c])
        )

        if denom <= 0:
            raise RuntimeError(
                "Invalid correlation denominator"
            )

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

    idx = int(
        np.nanargmax(local_joint)
    )

    return (
        int(local_lags[idx]),
        float(local_joint[idx]),
    )


def align_for_lag(a, b, lag):

    if lag == 0:
        return a, b

    if lag > 0:
        return (
            a[:-lag],
            b[lag:],
        )

    k = -lag

    return (
        a[k:],
        b[:-k],
    )


def verify_pair(a, b):

    lag, lag_score = estimate_common_lag(
        a,
        b,
    )

    aa, bb = align_for_lag(
        a,
        b,
        lag,
    )

    za = zscore_channels(aa)
    zb = zscore_channels(bb)

    correlations = []

    for c in range(3):

        r = np.corrcoef(
            za[:, c],
            zb[:, c],
        )[0, 1]

        correlations.append(
            float(r)
        )

    residual = za - zb

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    std_a = np.std(
        aa,
        axis=0,
    )

    std_b = np.std(
        bb,
        axis=0,
    )

    ratio = std_b / std_a

    mean_corr = float(
        np.mean(correlations)
    )

    min_corr = float(
        np.min(correlations)
    )

    max_corr = float(
        np.max(correlations)
    )

    spread = float(
        max_corr - min_corr
    )

    return {
        "common_lag_samples": lag,
        "common_lag_seconds": lag / FS,
        "abs_common_lag_samples": abs(lag),
        "lag_score": lag_score,

        "matched_samples": int(
            len(aa)
        ),

        "corr_X": correlations[0],
        "corr_Y": correlations[1],
        "corr_Z": correlations[2],

        "mean_channel_corr": mean_corr,
        "min_channel_corr": min_corr,
        "max_channel_corr": max_corr,
        "channel_corr_spread": spread,

        "joint_standardized_rmse": rmse,
        "joint_standardized_mae": mae,

        "std_ratio_X_b_over_a": float(
            ratio[0]
        ),

        "std_ratio_Y_b_over_a": float(
            ratio[1]
        ),

        "std_ratio_Z_b_over_a": float(
            ratio[2]
        ),
    }


def main():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    candidates = pd.read_csv(
        CANDIDATES_PATH
    )

    if len(manifest) != 96:
        raise RuntimeError(
            f"Expected 96 manifest rows; got {len(manifest)}"
        )

    if len(candidates) != 363:
        raise RuntimeError(
            f"Expected frozen candidate set of 363; "
            f"got {len(candidates)}"
        )

    manifest_map = manifest.set_index(
        "recording_id"
    ).to_dict(
        "index"
    )

    needed_ids = sorted(
        set(
            candidates["recording_a"]
        )
        |
        set(
            candidates["recording_b"]
        )
    )

    print("=" * 72)
    print(
        "EPSON LONG-SEGMENT COMMON-LAG VERIFICATION"
    )
    print("=" * 72)
    print(
        "Frozen candidates:",
        len(candidates),
    )
    print(
        "Unique recordings required:",
        len(needed_ids),
    )
    print(
        "Verification segment:",
        SEGMENT_LENGTH,
        "samples =",
        SEGMENT_LENGTH / FS,
        "s",
    )
    print(
        "Maximum common lag:",
        MAX_LAG,
        "samples =",
        MAX_LAG / FS,
        "s",
    )
    print(
        "Segment selection: fixed middle 10 seconds"
    )
    print(
        "No independent channel lag."
    )
    print(
        "No automatic deletion or split modification."
    )
    print()

    # --------------------------------------------------------
    # Read each required CSV ONCE and retain only its
    # middle 10-second segment.
    #
    # ~96 x 30000 x 3 float64 ≈ 69 MB.
    # --------------------------------------------------------

    segments = {}
    segment_meta = {}

    for idx, recording_id in enumerate(
        needed_ids,
        start=1,
    ):

        row = manifest_map[
            recording_id
        ]

        path = Path(
            row["filepath"]
        )

        print(
            f"[LOAD {idx:02d}/{len(needed_ids)}] "
            f"{recording_id}"
        )

        arr = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or
            arr.shape[1] != 3
        ):
            raise RuntimeError(
                f"{recording_id}: bad shape "
                f"{arr.shape}"
            )

        if not np.isfinite(
            arr
        ).all():
            raise RuntimeError(
                f"{recording_id}: "
                "nonfinite raw values"
            )

        seg, start, end = (
            get_middle_segment(
                arr
            )
        )

        segments[
            recording_id
        ] = seg.copy()

        segment_meta[
            recording_id
        ] = {
            "start": start,
            "end": end,
            "n_samples": int(
                len(arr)
            ),
        }

    print()
    print(
        "All required middle segments loaded."
    )
    print()

    results = []

    total = len(
        candidates
    )

    for i, row in candidates.iterrows():

        a_id = row[
            "recording_a"
        ]

        b_id = row[
            "recording_b"
        ]

        if (
            i == 0
            or
            (i + 1) % 25 == 0
            or
            i + 1 == total
        ):
            print(
                f"[VERIFY {i+1:03d}/{total}]"
            )

        metrics = verify_pair(
            segments[a_id],
            segments[b_id],
        )

        result = {
            "recording_a": a_id,
            "recording_b": b_id,

            "split_a": row[
                "split_a"
            ],

            "split_b": row[
                "split_b"
            ],

            "cross_split": bool(
                row["cross_split"]
            ),

            "class_id_a": int(
                row["class_id_a"]
            ),

            "class_id_b": int(
                row["class_id_b"]
            ),

            "config_a": row[
                "config_a"
            ],

            "config_b": row[
                "config_b"
            ],

            "feature_distance": float(
                row[
                    "feature_distance"
                ]
            ),

            "selection_reason": row[
                "selection_reason"
            ],

            "a_segment_start": (
                segment_meta[
                    a_id
                ]["start"]
            ),

            "a_segment_end_exclusive": (
                segment_meta[
                    a_id
                ]["end"]
            ),

            "b_segment_start": (
                segment_meta[
                    b_id
                ]["start"]
            ),

            "b_segment_end_exclusive": (
                segment_meta[
                    b_id
                ]["end"]
            ),

            **metrics,

            "verification_status": (
                "MEASURED_PENDING_REVIEW"
            ),

            "automatic_decision": (
                "NONE"
            ),
        }

        results.append(
            result
        )

    out = pd.DataFrame(
        results
    )

    # Ranking is for HUMAN REVIEW ONLY.
    # It is not a deletion/duplicate threshold.

    out = out.sort_values(
        [
            "mean_channel_corr",
            "joint_standardized_rmse",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    out[
        "review_rank"
    ] = np.arange(
        1,
        len(out) + 1,
    )

    out.to_csv(
        VERIFY_OUT,
        index=False,
    )

    cross = out[
        out["cross_split"]
        ==
        True
    ].copy()

    top_all = out.head(
        20
    )

    top_cross = cross.head(
        30
    )

    summary = {
        "status": "MEASUREMENT_COMPLETE_REVIEW_REQUIRED",

        "candidate_count": int(
            len(out)
        ),

        "cross_split_candidate_count": int(
            len(cross)
        ),

        "segment_selection": (
            "fixed middle 30000 samples"
        ),

        "segment_length_samples": (
            SEGMENT_LENGTH
        ),

        "segment_duration_seconds": (
            SEGMENT_LENGTH / FS
        ),

        "maximum_common_lag_samples": (
            MAX_LAG
        ),

        "maximum_common_lag_seconds": (
            MAX_LAG / FS
        ),

        "common_lag_for_all_channels": True,

        "independent_channel_lag": False,

        "automatic_duplicate_threshold": None,

        "automatic_delete": False,

        "automatic_split_change": False,

        "ranking_note": (
            "Sorted by mean three-channel correlation "
            "descending, then standardized RMSE ascending. "
            "Ranking is only for review."
        ),

        "top_pair": (
            out.iloc[0][
                "recording_a"
            ]
            + " <-> "
            + out.iloc[0][
                "recording_b"
            ]
        ),

        "top_pair_mean_corr": float(
            out.iloc[0][
                "mean_channel_corr"
            ]
        ),

        "top_pair_standardized_rmse": float(
            out.iloc[0][
                "joint_standardized_rmse"
            ]
        ),

        "important_limitations": [
            (
                "Verification uses one fixed middle "
                "10-second segment per recording."
            ),
            (
                "Passing does not establish "
                "session independence."
            ),
            (
                "Periodic signals may correlate strongly "
                "without sharing recording provenance."
            ),
            (
                "No pair is classified as duplicate "
                "solely from correlation."
            ),
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

    with REVIEW_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n\n"
            "===== LONG-SEGMENT VERIFICATION =====\n\n"
        )

        f.write(
            "Frozen candidate pairs measured: "
            f"{len(out)}\n"
        )

        f.write(
            "Cross-split candidate pairs measured: "
            f"{len(cross)}\n"
        )

        f.write(
            "Segment selection: fixed middle "
            "10 seconds from each recording.\n"
        )

        f.write(
            "Lag rule: one common lag across X/Y/Z, "
            f"limited to +/- {MAX_LAG} samples.\n"
        )

        f.write(
            "No automatic duplicate threshold was applied.\n"
        )

        f.write(
            "No recording was deleted or reassigned.\n"
        )

        f.write(
            "Status: MEASUREMENT_COMPLETE_REVIEW_REQUIRED\n"
        )

    cols = [
        "review_rank",
        "recording_a",
        "recording_b",
        "split_a",
        "split_b",
        "class_id_a",
        "class_id_b",
        "common_lag_samples",
        "corr_X",
        "corr_Y",
        "corr_Z",
        "mean_channel_corr",
        "joint_standardized_rmse",
        "joint_standardized_mae",
        "feature_distance",
    ]

    print()
    print("=" * 72)
    print(
        "VERIFICATION MEASUREMENT COMPLETE"
    )
    print("=" * 72)

    print(
        "Pairs verified       :",
        len(out),
    )

    print(
        "Cross-split verified :",
        len(cross),
    )

    print()
    print(
        "TOP 20 ALL CANDIDATES BY LONG-SEGMENT SIMILARITY"
    )
    print()

    print(
        top_all[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 30 CROSS-SPLIT BY LONG-SEGMENT SIMILARITY"
    )
    print()

    print(
        top_cross[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        VERIFY_OUT,
    )

    print(
        "Saved:",
        SUMMARY_OUT,
    )


if __name__ == "__main__":
    main()
