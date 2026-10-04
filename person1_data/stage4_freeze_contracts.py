from pathlib import Path
from datetime import datetime
import hashlib
import json
import pandas as pd

ROOT = Path(r"D:\DENSO_FACTORY\person1_data")

AUDIT = ROOT / "results" / "audit_v1"
SPLIT = ROOT / "results" / "split_seed42"
PREP = ROOT / "results" / "prepared_v1"
QA = ROOT / "results" / "qa_v2"

MANIFEST_PATH = AUDIT / "public_manifest.csv"
SPLIT_PATH = SPLIT / "split.csv"
PREP_PATH = PREP / "preparation_report.json"
STAGE3_PATH = QA / "near_duplicate_stage3_final.json"

DATA_OUT = QA / "data_contract.json"
ACCESS_OUT = QA / "access_and_scarcity.csv"
PROTOCOL_OUT = QA / "protocol_frozen.txt"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


# ============================================================
# PROTECT FROZEN OUTPUTS
# ============================================================

for p in [DATA_OUT, ACCESS_OUT, PROTOCOL_OUT]:
    if p.exists():
        raise RuntimeError(
            f"STOP: frozen output already exists: {p}"
        )


# ============================================================
# LOAD
# ============================================================

manifest = pd.read_csv(MANIFEST_PATH)
split = pd.read_csv(SPLIT_PATH)

with PREP_PATH.open("r", encoding="utf-8") as f:
    prep = json.load(f)

with STAGE3_PATH.open("r", encoding="utf-8") as f:
    stage3 = json.load(f)


# ============================================================
# CORE VALIDATION
# ============================================================

assert len(manifest) == 96
assert manifest["recording_id"].nunique() == 96

assert len(split) == 96
assert split["recording_id"].nunique() == 96

assert prep["status"] == "complete"

assert stage3["next_stage_allowed"] is True
assert stage3["confirmed_overlap_pairs"] == 0


# split.csv already contains class metadata.
# Use it directly rather than duplicating the same columns
# through a merge that would create _x / _y suffixes.

meta = split.copy()

required_meta_cols = [
    "recording_id",
    "split",
    "class_id",
    "config_label",
    "binary_label",
]

missing_cols = [
    c for c in required_meta_cols
    if c not in meta.columns
]

if missing_cols:
    raise RuntimeError(
        f"Missing required split metadata columns: {missing_cols}"
    )

if meta[required_meta_cols].isna().any().any():
    raise RuntimeError(
        "Required split metadata contain missing values"
    )

# Independently verify that the class metadata already
# stored in split.csv agree with public_manifest.csv.

consistency = meta[
    [
        "recording_id",
        "class_id",
        "config_label",
        "binary_label",
    ]
].merge(
    manifest[
        [
            "recording_id",
            "class_id",
            "config_label",
            "binary_label",
        ]
    ],
    on="recording_id",
    how="left",
    suffixes=("_split", "_manifest"),
    validate="one_to_one",
)

if len(consistency) != 96:
    raise RuntimeError(
        f"Expected 96 metadata consistency rows, got {len(consistency)}"
    )

for col in [
    "class_id",
    "config_label",
    "binary_label",
]:
    left = consistency[f"{col}_split"].astype(str)
    right = consistency[f"{col}_manifest"].astype(str)

    bad = left != right

    if bad.any():
        bad_ids = consistency.loc[
            bad,
            "recording_id"
        ].tolist()

        raise RuntimeError(
            f"Split/manifest mismatch for {col}: {bad_ids}"
        )

print(
    "Split metadata consistency with manifest: PASS"
)


expected_split = {
    "train": 48,
    "validation": 24,
    "test": 24,
}

actual_split = (
    meta.groupby("split")
    .size()
    .to_dict()
)

if actual_split != expected_split:
    raise RuntimeError(
        f"Unexpected split counts: {actual_split}"
    )


for cid in range(6):

    counts = (
        meta[
            meta["class_id"] == cid
        ]
        .groupby("split")
        .size()
        .to_dict()
    )

    expected = {
        "train": 8,
        "validation": 4,
        "test": 4,
    }

    if counts != expected:
        raise RuntimeError(
            f"Class {cid} split counts wrong: {counts}"
        )


train_ids = set(
    meta.loc[
        meta["split"] == "train",
        "recording_id"
    ]
)

val_ids = set(
    meta.loc[
        meta["split"] == "validation",
        "recording_id"
    ]
)

test_ids = set(
    meta.loc[
        meta["split"] == "test",
        "recording_id"
    ]
)

if train_ids & val_ids:
    raise RuntimeError("Train/validation overlap")

if train_ids & test_ids:
    raise RuntimeError("Train/test overlap")

if val_ids & test_ids:
    raise RuntimeError("Validation/test overlap")


# ============================================================
# NORMAL-ONLY
# ============================================================

normal = prep["normal_only"]

normal_ids = set(
    normal["train_ids"]
)

if len(normal_ids) != 8:
    raise RuntimeError("Normal-only pool must have 8 records")

if not normal_ids <= train_ids:
    raise RuntimeError("Normal-only pool contains non-train IDs")

normal_meta = meta[
    meta["recording_id"].isin(normal_ids)
]

if not (
    normal_meta["class_id"] == 0
).all():
    raise RuntimeError(
        "Normal-only pool contains fault class"
    )

if set(
    normal["scaler"]["fit_recording_ids"]
) != normal_ids:
    raise RuntimeError(
        "Normal-only scaler fit IDs are incorrect"
    )


# ============================================================
# SCARCITY BUDGETS
# ============================================================

expected = {
    "1": {
        "total": 13,
        "fault": 5,
        "per_fault_class": 1,
        "windows": 1222,
    },

    "2": {
        "total": 18,
        "fault": 10,
        "per_fault_class": 2,
        "windows": 1692,
    },

    "4": {
        "total": 28,
        "fault": 20,
        "per_fault_class": 4,
        "windows": 2632,
    },
}

budget_sets = {}
budget_info = {}

for k in ["1", "2", "4"]:

    b = prep["budgets"][k]
    exp = expected[k]

    ids = set(b["train_ids"])
    fault_ids = set(b["fault_train_ids"])
    scaler_ids = set(
        b["scaler"]["fit_recording_ids"]
    )

    budget_sets[k] = ids

    if len(ids) != exp["total"]:
        raise RuntimeError(
            f"k{k}: incorrect train record count"
        )

    if len(fault_ids) != exp["fault"]:
        raise RuntimeError(
            f"k{k}: incorrect fault record count"
        )

    if not normal_ids <= ids:
        raise RuntimeError(
            f"k{k}: does not contain all 8 normal records"
        )

    if ids - normal_ids != fault_ids:
        raise RuntimeError(
            f"k{k}: fault membership mismatch"
        )

    if not ids <= train_ids:
        raise RuntimeError(
            f"k{k}: contains validation/test IDs"
        )

    if scaler_ids != ids:
        raise RuntimeError(
            f"k{k}: scaler IDs != permitted train IDs"
        )

    if scaler_ids & val_ids or scaler_ids & test_ids:
        raise RuntimeError(
            f"k{k}: scaler leakage"
        )

    bmeta = meta[
        meta["recording_id"].isin(ids)
    ]

    fault_counts = (
        bmeta[
            bmeta["class_id"] != 0
        ]
        .groupby("class_id")
        .size()
        .to_dict()
    )

    for cid in range(1, 6):

        if fault_counts.get(cid, 0) != exp["per_fault_class"]:
            raise RuntimeError(
                f"k{k}: fault class {cid} count incorrect"
            )

    wc = b["window_counts"]

    if int(wc["train"]) != exp["windows"]:
        raise RuntimeError(
            f"k{k}: train window count incorrect"
        )

    if int(wc["validation"]) != 2256:
        raise RuntimeError(
            f"k{k}: validation window count incorrect"
        )

    if int(wc["test"]) != 2256:
        raise RuntimeError(
            f"k{k}: test window count incorrect"
        )

    budget_info[f"k{k}"] = {
        "fault_records_per_fault_class":
            exp["per_fault_class"],

        "normal_train_records": 8,
        "fault_train_records": exp["fault"],
        "total_train_records": exp["total"],

        "train_window_count":
            int(wc["train"]),

        "validation_window_count":
            int(wc["validation"]),

        "test_window_count":
            int(wc["test"]),

        "train_recording_ids":
            sorted(ids),

        "fault_train_recording_ids":
            sorted(fault_ids),

        "scaler": {
            "fit_recording_ids":
                sorted(scaler_ids),

            "mean":
                b["scaler"]["mean"],

            "std":
                b["scaler"]["std"],

            "ddof":
                b["scaler"]["ddof"],

            "weighting":
                b["scaler"]["weighting"],
        },
    }


if not (
    budget_sets["1"]
    <
    budget_sets["2"]
    <
    budget_sets["4"]
):
    raise RuntimeError(
        "Budgets are not strictly nested k1 < k2 < k4"
    )


unused_k4 = sorted(
    train_ids
    -
    budget_sets["4"]
)

if len(unused_k4) != 20:
    raise RuntimeError(
        f"Expected 20 unused train records under k4; got {len(unused_k4)}"
    )

unused_meta = meta[
    meta["recording_id"].isin(
        unused_k4
    )
]

if not (
    unused_meta["class_id"] != 0
).all():
    raise RuntimeError(
        "Unused k4 pool unexpectedly contains normal records"
    )


# ============================================================
# VALIDATION / TEST COUNTS
# ============================================================

val_meta = meta[
    meta["split"] == "validation"
]

test_meta = meta[
    meta["split"] == "test"
]

val_normal = int(
    (val_meta["class_id"] == 0).sum()
)

val_fault = int(
    (val_meta["class_id"] != 0).sum()
)

test_normal = int(
    (test_meta["class_id"] == 0).sum()
)

test_fault = int(
    (test_meta["class_id"] != 0).sum()
)

if (val_normal, val_fault) != (4, 20):
    raise RuntimeError(
        "Validation composition incorrect"
    )

if (test_normal, test_fault) != (4, 20):
    raise RuntimeError(
        "Test composition incorrect"
    )

if prep["labeled_validation_records"] != 20:
    raise RuntimeError(
        "labeled_validation_records must equal 20"
    )


# ============================================================
# CLASS MAP
# ============================================================

class_map = {}

for cid in range(6):

    rows = manifest[
        manifest["class_id"] == cid
    ]

    configs = (
        rows["config_label"]
        .astype(str)
        .unique()
        .tolist()
    )

    binary = (
        rows["binary_label"]
        .astype(int)
        .unique()
        .tolist()
    )

    if len(configs) != 1 or len(binary) != 1:
        raise RuntimeError(
            f"Inconsistent mapping for class {cid}"
        )

    class_map[str(cid)] = {
        "config_label": configs[0],
        "binary_label": int(binary[0]),
        "record_count": int(len(rows)),
    }


# ============================================================
# DATA CONTRACT
# ============================================================

created = (
    datetime.now()
    .astimezone()
    .isoformat()
)

contract = {
    "contract_version": "data_contract_v1",
    "status": "FROZEN",
    "created_at_local": created,

    "dataset": {
        "name": "Epson trimmed",
        "record_count": 96,
        "split_unit": "recording/file",
        "session_independence_verified": False,
        "single_rig": True,
        "nominal_rpm": 1200,
    },

    "signal": {
        "sampling_rate_hz": 3000,
        "channels": ["X", "Y", "Z"],
        "raw_unit": "mm/s",
        "prepared_unit": "dimensionless z-score",

        "window_length_samples": 3000,
        "window_duration_seconds": 1.0,

        "stride_samples": 3000,
        "stride_seconds": 1.0,
    },

    "class_mapping": class_map,

    "split": {
        "seed": 42,

        "train_records": 48,
        "validation_records": 24,
        "test_records": 24,

        "records_per_class": {
            "train": 8,
            "validation": 4,
            "test": 4,
        },

        "recording_id_overlap": False,
    },

    "scarcity": {
        "selection_seed": prep["seed"],
        "nested": True,

        "fixed_normal_train_records": 8,

        "budgets": budget_info,

        "unused_train_fault_records_under_k4":
            20,

        "unused_train_recording_ids_under_k4":
            unused_k4,
    },

    "normal_only": {
        "train_record_count": 8,

        "train_recording_ids":
            sorted(normal_ids),

        "train_windows":
            normal["window_counts"]["train"],

        "validation_windows":
            normal["window_counts"]["validation"],

        "test_windows":
            normal["window_counts"]["test"],

        "scaler": normal["scaler"],
    },

    "normalization": {
        "method": "per-channel z-score",
        "ddof": 0,

        "budget_specific_scaler": True,

        "fit_scope": (
            "all samples in allowed training records, "
            "including unused tail"
        ),

        "validation_used_for_scaler_fit": False,
        "test_used_for_scaler_fit": False,
    },

    "validation_policy": {
        "record_count": 24,
        "normal_records": 4,
        "fault_records": 20,

        "allowed": [
            "model selection",
            "early stopping",
            "threshold selection",
            "hyperparameter selection",
        ],

        "forbidden": [
            "scaler fitting",
            "generator weight fitting",
            "detector weight fitting",
            "final held-out performance claim",
        ],
    },

    "test_policy": {
        "record_count": 24,
        "normal_records": 4,
        "fault_records": 20,

        "allowed": [
            "final held-out evaluation after protocol freeze",
            "pre-model frozen data-integrity audit only"
        ],

        "forbidden": [
            "scaler fitting",
            "generator fitting",
            "detector fitting",
            "threshold selection",
            "hyperparameter selection",
            "preprocessing selection",
            "model-selection decisions",
        ],
    },

    "integrity_gate": {
        "stage3_status":
            stage3["status"],

        "confirmed_overlap_pairs":
            stage3["confirmed_overlap_pairs"],

        "split_action":
            stage3["split_action"],

        "record_deletion":
            stage3["record_deletion"],
    },

    "limitations": [
        "Recording-session independence is unverified.",
        "Dataset represents one rig at one nominal speed.",
        "Near-duplicate review cannot prove complete source independence.",
        "Per-sample timestamps are unavailable.",
    ],

    "change_control": {
        "silent_changes_allowed": False,

        "rule": (
            "Any change to split, scarcity membership, "
            "windowing, normalization, access policy or "
            "preprocessing requires a new version."
        ),
    },
}


with DATA_OUT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        contract,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# ACCESS / SCARCITY TABLE
# ============================================================

rows = []


def add_row(
    access_set,
    split_name,
    budget,
    ids,
    normal_n,
    fault_n,
    fault_per_class,
    windows,
    scaler,
    model,
    generator,
    threshold,
    final_metric,
    note,
):

    rows.append({
        "access_set": access_set,
        "split": split_name,
        "budget": budget,

        "record_count": len(ids),
        "normal_records": normal_n,
        "fault_records": fault_n,

        "fault_records_per_fault_class":
            fault_per_class,

        "window_count": windows,

        "scaler_fit_allowed": scaler,
        "model_fit_allowed": model,
        "generator_fit_allowed": generator,

        "threshold_selection_allowed":
            threshold,

        "final_metric_reporting_allowed":
            final_metric,

        "recording_ids":
            ";".join(sorted(ids)),

        "note": note,
    })


add_row(
    "normal_only_train",
    "train",
    "normal_only",
    normal_ids,
    8,
    0,
    0,
    normal["window_counts"]["train"],
    "YES",
    "YES",
    "NO",
    "NO",
    "NO",
    "Fixed 8 normal training recordings."
)


for k in ["1", "2", "4"]:

    info = budget_info[f"k{k}"]

    add_row(
        f"k{k}_train",
        "train",
        f"k{k}",
        set(info["train_recording_ids"]),
        8,
        info["fault_train_records"],
        info["fault_records_per_fault_class"],
        info["train_window_count"],
        "YES",
        "YES",
        "YES",
        "NO",
        "NO",
        "Frozen scarcity pool."
    )


add_row(
    "validation_fixed",
    "validation",
    "fixed",
    val_ids,
    4,
    20,
    4,
    2256,
    "NO",
    "NO",
    "NO",
    "YES",
    "NO",
    "Selection/early stopping/threshold/hyperparameters only."
)


add_row(
    "test_final",
    "test",
    "fixed",
    test_ids,
    4,
    20,
    4,
    2256,
    "NO",
    "NO",
    "NO",
    "NO",
    "YES",
    "Final held-out evaluation only; integrity audit exception is pre-model and frozen."
)


access = pd.DataFrame(rows)

access.to_csv(
    ACCESS_OUT,
    index=False
)


# ============================================================
# FROZEN PROTOCOL
# ============================================================

hashes = {
    "public_manifest.csv":
        sha256(MANIFEST_PATH),

    "split.csv":
        sha256(SPLIT_PATH),

    "preparation_report.json":
        sha256(PREP_PATH),

    "near_duplicate_stage3_final.json":
        sha256(STAGE3_PATH),

    "data_contract.json":
        sha256(DATA_OUT),

    "access_and_scarcity.csv":
        sha256(ACCESS_OUT),
}


text = f"""EPSON DATA PROTOCOL - FROZEN V1
========================================================================

Frozen at:
{created}

STATUS: FROZEN

1. DATASET
- Epson trimmed dataset.
- 96 recordings.
- Six classes, 16 recordings per class.
- Fs = 3000 Hz.
- Channels = X, Y, Z.
- Raw quantity = vibration velocity in mm/s.
- Nominal speed = 1200 rpm.
- Session independence remains unverified.

2. SPLIT
- Split unit = recording/file.
- Seed = 42.
- Train = 48 records = 8/class.
- Validation = 24 records = 4/class.
- Test = 24 records = 4/class.
- Recording IDs are disjoint.

3. WINDOWING
- Window = 3000 samples = 1.0 s.
- Stride = 3000 samples = 1.0 s.
- Recording split occurs before windowing.

4. SCARCITY
- Selection seed = 101.
- Fixed normal pool = 8 train records.
- k1 = 8 normal + 5 fault = 13.
- k2 = 8 normal + 10 fault = 18.
- k4 = 8 normal + 20 fault = 28.
- k1 is a strict subset of k2.
- k2 is a strict subset of k4.
- 20 remaining train fault records are excluded from scarcity runs.

5. NORMALIZATION
- Per-channel z-score.
- ddof = 0.
- Separate scaler for normal-only, k1, k2 and k4.
- Scaler is fit only on permitted train records.
- All samples in permitted records contribute, including unused tail.
- Validation/test never fit the scaler.

6. VALIDATION
- May be used for model selection, early stopping,
  threshold selection and hyperparameter selection.
- Must not fit scaler, generator weights or detector weights.
- Must not be reported as final held-out performance.

7. TEST
- Used for final held-out evaluation only after protocol freeze.
- Exception: frozen pre-model data-integrity auditing is allowed.
- Test must not guide preprocessing/model/threshold/hyperparameter choices.

8. INTEGRITY
- Stage 2 signal QA completed.
- Stage 3 near/shifted duplicate review:
  PASS WITH OPEN-LIMITATION.
- Confirmed recording overlap pairs = 0.
- No record deletion.
- No split redesign required.

9. LIMITATIONS
- Session independence unverified.
- Single rig / nominal speed.
- Near-duplicate analysis cannot prove complete source independence.
- Per-sample timestamps unavailable.

10. CHANGE CONTROL
- No silent change to split, scarcity membership, windowing,
  normalization, preprocessing or access policy.
- Any such change requires a new protocol/prepared-data version.

11. SHA256
"""

for name, digest in hashes.items():
    text += f"- {name}: {digest}\n"

text += "\nEND OF FROZEN PROTOCOL\n"

PROTOCOL_OUT.write_text(
    text,
    encoding="utf-8"
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 72)
print("STAGE 4 CONTRACT GENERATION")
print("=" * 72)

print()
print("Core validation: PASS")
print("Split counts:", actual_split)

print(
    "Normal-only records:",
    len(normal_ids)
)

print(
    "k1/k2/k4:",
    len(budget_sets["1"]),
    len(budget_sets["2"]),
    len(budget_sets["4"])
)

print(
    "Nested:",
    budget_sets["1"]
    <
    budget_sets["2"]
    <
    budget_sets["4"]
)

print(
    "Validation normal/fault:",
    val_normal,
    "/",
    val_fault
)

print(
    "Test normal/fault:",
    test_normal,
    "/",
    test_fault
)

print(
    "Unused train fault records under k4:",
    len(unused_k4)
)

print()
print("========== ACCESS AND SCARCITY ==========")

print(
    access[
        [
            "access_set",
            "split",
            "budget",
            "record_count",
            "normal_records",
            "fault_records",
            "fault_records_per_fault_class",
            "window_count",
            "scaler_fit_allowed",
            "model_fit_allowed",
            "generator_fit_allowed",
            "threshold_selection_allowed",
            "final_metric_reporting_allowed",
        ]
    ].to_string(index=False)
)

print()
print("========== CREATED FILES ==========")

for p in [
    DATA_OUT,
    ACCESS_OUT,
    PROTOCOL_OUT,
]:
    print(
        p.name,
        "| bytes=",
        p.stat().st_size,
        "| sha256=",
        sha256(p)
    )

print()
print("========== STAGE 4 GENERATION COMPLETE ==========")


if __name__ == "__main__":
    pass
