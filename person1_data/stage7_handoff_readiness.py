from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import numpy as np


ROOT = Path(r"D:\DENSO_FACTORY\person1_data")
QA = ROOT / "results" / "qa_v2"

sys.path.insert(
    0,
    str(ROOT)
)

from loader_v1 import EpsonPreparedLoader


CONTRACT = QA / "data_contract.json"
ACCESS = QA / "access_and_scarcity.csv"
DATASET_CARD = QA / "dataset_card.txt"
SOURCE_REGISTER = QA / "source_register.csv"

STAGE3 = QA / "near_duplicate_stage3_final.json"
LOADER_TEST = QA / "loader_selftest.json"
STAGE6 = QA / "stage6_final.json"
REPRO_FINAL = QA / "stage6_reproduction_final.json"
REPRO_REPORT = QA / "reproducibility_report.json"
PACKAGE_MANIFEST = QA / "package_manifest.csv"

PREP_REPORT = (
    ROOT
    / "results"
    / "prepared_v1"
    / "preparation_report.json"
)

SPLIT_CSV = (
    ROOT
    / "results"
    / "split_seed42"
    / "split.csv"
)

SMOKE_OUT = QA / "consumer_smoke_test.json"
README_OUT = QA / "HANDOFF_README.txt"
SMOKE_GUIDE_OUT = QA / "RECIPIENT_SMOKE_TEST.txt"
ACCEPTANCE_OUT = QA / "RECIPIENT_ACCEPTANCE_TEMPLATE.txt"
CHECKLIST_OUT = QA / "HANDOFF_CHECKLIST.txt"
MANIFEST_OUT = QA / "handoff_manifest.json"
FINAL_OUT = QA / "stage7_readiness.json"


def read_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


# ============================================================
# DO NOT OVERWRITE HANDOFF ARTIFACTS
# ============================================================

for p in [
    SMOKE_OUT,
    README_OUT,
    SMOKE_GUIDE_OUT,
    ACCEPTANCE_OUT,
    CHECKLIST_OUT,
    MANIFEST_OUT,
    FINAL_OUT,
]:

    if p.exists():

        raise RuntimeError(
            f"STOP: Stage-7 output already exists: {p}"
        )


# ============================================================
# REQUIRED EVIDENCE
# ============================================================

required = [
    CONTRACT,
    ACCESS,
    DATASET_CARD,
    SOURCE_REGISTER,
    STAGE3,
    LOADER_TEST,
    STAGE6,
    REPRO_FINAL,
    REPRO_REPORT,
    PACKAGE_MANIFEST,
    PREP_REPORT,
    SPLIT_CSV,
]

missing = [
    str(p)
    for p in required
    if not p.exists()
]

if missing:

    raise RuntimeError(
        "Missing handoff evidence:\n"
        + "\n".join(missing)
    )


contract = read_json(
    CONTRACT
)

stage3 = read_json(
    STAGE3
)

loader_report = read_json(
    LOADER_TEST
)

stage6 = read_json(
    STAGE6
)

repro_final = read_json(
    REPRO_FINAL
)

repro_report = read_json(
    REPRO_REPORT
)

prep = read_json(
    PREP_REPORT
)


# ============================================================
# GATES FROM PRIOR STAGES
# ============================================================

if contract.get("status") != "FROZEN":
    raise RuntimeError(
        "Data contract is not frozen"
    )

if stage3.get(
    "confirmed_overlap_pairs"
) != 0:
    raise RuntimeError(
        "Confirmed cross-split recording overlap exists"
    )

if stage3.get(
    "next_stage_allowed"
) is not True:
    raise RuntimeError(
        "Stage 3 does not permit handoff"
    )

if loader_report.get(
    "status"
) != "PASS":
    raise RuntimeError(
        "Loader self-test not PASS"
    )

if stage6.get(
    "next_stage_allowed"
) is not True:
    raise RuntimeError(
        "Stage 6 package gate not satisfied"
    )

if repro_final.get(
    "next_stage_allowed"
) is not True:
    raise RuntimeError(
        "Independent reproduction gate not satisfied"
    )

if repro_final.get(
    "array_content_exact"
) is not True:
    raise RuntimeError(
        "Reproduced arrays are not exact"
    )

if repro_final.get(
    "scalers_exact"
) is not True:
    raise RuntimeError(
        "Reproduced scalers are not exact"
    )

if repro_final.get(
    "window_metadata_exact"
) is not True:
    raise RuntimeError(
        "Reproduced window metadata are not exact"
    )


# ============================================================
# CONSUMER-SIDE SMOKE TEST THROUGH PUBLIC LOADER API
# ============================================================

loader = EpsonPreparedLoader(
    ROOT
)

k1 = loader.load(
    "k1",
    "train",
)

k2 = loader.load(
    "k2",
    "train",
)

k4 = loader.load(
    "k4",
    "train",
)

normal_only = loader.load(
    "normal_only",
    "train",
)


# ----- k1 expected shape / labels -----

if k1.X.shape != (
    1222,
    3,
    3000,
):
    raise RuntimeError(
        f"Unexpected k1 shape: {k1.X.shape}"
    )

if sorted(
    np.unique(
        k1.y_class
    ).tolist()
) != [
    0, 1, 2, 3, 4, 5
]:
    raise RuntimeError(
        "Unexpected k1 class labels"
    )

if sorted(
    np.unique(
        k1.y_binary
    ).tolist()
) != [
    0, 1
]:
    raise RuntimeError(
        "Unexpected binary labels"
    )

if not np.array_equal(
    k1.y_binary,
    (
        k1.y_class != 0
    ).astype(
        np.int64
    ),
):
    raise RuntimeError(
        "y_binary disagrees with y_class"
    )


# ----- recording counts -----

record_counts = {
    "normal_only":
        len(
            set(
                normal_only.recording_id.tolist()
            )
        ),

    "k1":
        len(
            set(
                k1.recording_id.tolist()
            )
        ),

    "k2":
        len(
            set(
                k2.recording_id.tolist()
            )
        ),

    "k4":
        len(
            set(
                k4.recording_id.tolist()
            )
        ),
}

if record_counts != {
    "normal_only": 8,
    "k1": 13,
    "k2": 18,
    "k4": 28,
}:

    raise RuntimeError(
        f"Unexpected recording counts: {record_counts}"
    )


# ----- budget nesting -----

k1_ids = set(
    prep[
        "budgets"
    ][
        "1"
    ][
        "train_ids"
    ]
)

k2_ids = set(
    prep[
        "budgets"
    ][
        "2"
    ][
        "train_ids"
    ]
)

k4_ids = set(
    prep[
        "budgets"
    ][
        "4"
    ][
        "train_ids"
    ]
)

if not (
    k1_ids
    <
    k2_ids
    <
    k4_ids
):
    raise RuntimeError(
        "k1 < k2 < k4 nesting failed"
    )


# ----- frozen scaler membership -----

scaler = loader.get_scaler(
    "k1"
)

scaler_ids = set(
    scaler[
        "fit_recording_ids"
    ]
)

if scaler_ids != k1_ids:
    raise RuntimeError(
        "k1 scaler IDs differ from frozen k1 train IDs"
    )


split_table = loader.split_table

validation_ids = set(
    split_table.loc[
        split_table[
            "split"
        ] == "validation",
        "recording_id",
    ].astype(
        str
    )
)

test_ids = set(
    split_table.loc[
        split_table[
            "split"
        ] == "test",
        "recording_id",
    ].astype(
        str
    )
)

if scaler_ids & validation_ids:
    raise RuntimeError(
        "Validation leakage into k1 scaler"
    )

if scaler_ids & test_ids:
    raise RuntimeError(
        "Test leakage into k1 scaler"
    )


# ----- normal only -----

if np.any(
    normal_only.y_class != 0
):
    raise RuntimeError(
        "normal_only train contains fault"
    )


# ============================================================
# INVERSE-TRANSFORM SMOKE TEST
# ============================================================

mean = np.asarray(
    scaler[
        "mean"
    ],
    dtype=np.float64,
)

scale = np.asarray(
    scaler[
        "scale"
    ],
    dtype=np.float64,
)

x = k1.X[
    0
].T.astype(
    np.float64
)

inverse = (
    x
    *
    scale
    +
    mean
)

roundtrip = (
    (
        inverse
        -
        mean
    )
    /
    scale
).T

inverse_roundtrip_max_abs_error = float(
    np.max(
        np.abs(
            roundtrip
            -
            k1.X[
                0
            ].astype(
                np.float64
            )
        )
    )
)

if inverse_roundtrip_max_abs_error > 1e-10:

    raise RuntimeError(
        "Inverse-transform roundtrip failed: "
        f"{inverse_roundtrip_max_abs_error}"
    )


# ============================================================
# PRODUCER-SIDE SMOKE REPORT
# ============================================================

created = (
    datetime.now()
    .astimezone()
    .isoformat()
)


smoke = {
    "stage": 7,

    "test":
        "producer_handoff_consumer_api_smoke_test",

    "created_at_local":
        created,

    "status":
        "PASS",

    "important_scope":
        (
            "Technical handoff readiness only. "
            "This does NOT count as recipient acceptance."
        ),

    "k1": {
        "X_shape":
            list(
                k1.X.shape
            ),

        "X_dtype":
            str(
                k1.X.dtype
            ),

        "y_class_values":
            sorted(
                np.unique(
                    k1.y_class
                ).tolist()
            ),

        "y_binary_values":
            sorted(
                np.unique(
                    k1.y_binary
                ).tolist()
            ),

        "unique_recordings":
            record_counts[
                "k1"
            ],
    },

    "recording_counts":
        record_counts,

    "budget_nesting":
        "PASS",

    "k1_scaler_fit_record_count":
        len(
            scaler_ids
        ),

    "validation_in_k1_scaler":
        False,

    "test_in_k1_scaler":
        False,

    "normal_only_fault_windows":
        0,

    "inverse_transform_roundtrip_max_abs_error":
        inverse_roundtrip_max_abs_error,

    "stage6_reproduction_exact":
        True,
}


SMOKE_OUT.write_text(
    json.dumps(
        smoke,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# HANDOFF README
# ============================================================

readme = f"""EPSON PERSON 1 - HANDOFF README
========================================================================

Release:
EPSON_PERSON1_HANDOFF_V1

Created:
{created}

CURRENT STATE
-------------
Producer-side technical readiness: PASS
Recipient acceptance: PENDING
Overall handoff state:
READY_FOR_RECIPIENT_ACCEPTANCE_WITH_OPEN_LIMITATIONS

DATA INTERFACE
--------------
Use:

    from loader_v1 import EpsonPreparedLoader

    loader = EpsonPreparedLoader()
    data = loader.load("k1", "train")

X:
[N, 3, 3000], float32, z-score

y_class:
0..5 configuration label

y_binary:
0 normal / 1 abnormal

recording_id:
source recording/file identifier

start_sample:
window start within the trimmed recording

DO NOT
------
- Do not refit scaler on validation or test.
- Do not z-score each window independently.
- Do not change split seed42 or scarcity seed101 silently.
- Do not add unused train faults into k1/k2/k4 experiments.
- Do not use test for model, preprocessing, threshold or
  hyperparameter selection.
- Do not treat windows as independent physical sessions.
- Do not claim session-independent evaluation.
- Do not claim multi-speed evaluation.
- Do not claim failure-time / remaining-life prediction.
- Do not claim this is DENSO factory measurement data.

PERSON 2
--------
Must independently confirm:
1. k1/train loads through loader_v1.
2. y_class and y_binary are understood separately.
3. Frozen scaler is used; no refit on validation/test.
4. Same real-record subset is retained across compared methods.
5. Metrics/aggregation respect recording/group identity.
6. Validation selects thresholds/aggregation/model choices.
7. Test is final held-out evaluation only.

PERSON 3
--------
Must independently confirm:
1. Only permitted train IDs for the selected k are used.
2. No checkpoint trained with a larger-k real pool is reused
   as if it belonged to a smaller-k scarcity experiment.
3. Generator output contains all X/Y/Z channels jointly.
4. Condition is configuration/fault class; this dataset does
   not provide a variable-speed research condition.
5. Generated values are in the correct z-score space and use
   the matching frozen scaler when inverse transforming.
6. Synthetic records carry synthetic provenance metadata:
   source_type, generator/version, seed and condition.
7. Synthetic data must not pretend to have a real source recording.

PERSON 4
--------
Use these evidence artifacts:
- dataset_card.txt
- source_register.csv
- access_and_scarcity.csv
- data_contract.json
- protocol_frozen.txt
- loader_selftest.json
- reproducibility_report.json
- HANDOFF_CHECKLIST.txt

OPEN LIMITATIONS
----------------
- Recording-session independence is unverified.
- Per-sample timestamps are unavailable; timestamp jitter was
  not measured.
- Dataset is one rig / nominal speed.
- Near-duplicate checks reduce risk but cannot prove absolute
  source independence.
- Dataset redistribution/use permissions remain to be verified
  for the intended external/public/commercial scope.
- Git commit history was unavailable at release time.
- Independent reproduction was performed in a new output
  directory on the same machine/environment, not cross-machine.
- Recipient acceptance has not yet been performed.

CHANGE CONTROL
--------------
Any change to source data, split, window length/stride,
scarcity membership, scaler or preprocessing requires a new
version and a new downstream-impact statement.
"""

README_OUT.write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# RECIPIENT SMOKE TEST GUIDE
# ============================================================

smoke_guide = r"""EPSON RECIPIENT SMOKE TEST
========================================================================

Run from:

    D:\DENSO_FACTORY

PERSON 2 - MINIMUM TECHNICAL CHECK
----------------------------------

PowerShell:

python -c "import sys; sys.path.insert(0,r'D:\DENSO_FACTORY\person1_data'); from loader_v1 import EpsonPreparedLoader; import numpy as np; L=EpsonPreparedLoader(); d=L.load('k1','train'); print('X',d.X.shape,d.X.dtype); print('y_class',np.unique(d.y_class)); print('y_binary',np.unique(d.y_binary)); print('recordings',len(set(d.recording_id.tolist()))); print('scaler IDs',len(L.get_scaler('k1')['fit_recording_ids']))"

Expected:

X (1222, 3, 3000) float32
y_class [0 1 2 3 4 5]
y_binary [0 1]
recordings 13
scaler IDs 13

Then confirm in writing:
- I understand y_class != y_binary.
- I will not fit scaler on validation/test.
- I will use recording_id for grouped evaluation/aggregation.
- I will use validation for selection and test only at final evaluation.

PERSON 3 - MINIMUM TECHNICAL CHECK
----------------------------------

PowerShell:

python -c "import sys; sys.path.insert(0,r'D:\DENSO_FACTORY\person1_data'); from loader_v1 import EpsonPreparedLoader; L=EpsonPreparedLoader(); print('k1',len(L.get_scaler('k1')['fit_recording_ids'])); print('k2',len(L.get_scaler('k2')['fit_recording_ids'])); print('k4',len(L.get_scaler('k4')['fit_recording_ids'])); print('shape',L.load('k1','train').X.shape)"

Expected:

k1 13
k2 18
k4 28
shape (1222, 3, 3000)

Then confirm in writing:
- Generator will use only real train IDs permitted for its k.
- It will generate three channels jointly.
- Synthetic provenance will not impersonate real recordings.
- Matching frozen scaler will be stored with generated output.

PERSON 4 - DOCUMENT CHECK
-------------------------

Open:
- person1_data/results/qa_v2/dataset_card.txt
- person1_data/results/qa_v2/access_and_scarcity.csv
- person1_data/results/qa_v2/data_contract.json
- person1_data/results/qa_v2/reproducibility_report.txt
- person1_data/results/qa_v2/HANDOFF_CHECKLIST.txt

Confirm that presentation material does NOT claim:
- multi-speed evidence,
- failure-time prediction,
- session-independent evaluation,
- DENSO factory measurement data.

IMPORTANT
---------
A Person-1 run of these commands is only a readiness test.
Recipient acceptance must be recorded by the actual recipient.
"""

SMOKE_GUIDE_OUT.write_text(
    smoke_guide,
    encoding="utf-8",
)


# ============================================================
# RECIPIENT ACCEPTANCE TEMPLATE
# ============================================================

acceptance = """EPSON RECIPIENT ACCEPTANCE TEMPLATE
========================================================================

Release:
EPSON_PERSON1_HANDOFF_V1

This file must be completed by the actual recipient.
Person 1 must not self-sign these fields.

PERSON 2
--------
Checker:
Date:
Status: OPEN-LIMITATION (pending recipient check)

Checks:
[ ] k1/train loader output matches expected shape/labels.
[ ] y_class and y_binary distinction understood.
[ ] Frozen scaler policy understood.
[ ] Group/recording-level evaluation policy understood.
[ ] Validation/test access policy understood.

Remaining issue / comment:


PERSON 3
--------
Checker:
Date:
Status: OPEN-LIMITATION (pending recipient check)

Checks:
[ ] k1/k2/k4 permitted train pools understood.
[ ] No larger-k checkpoint leakage into smaller-k study.
[ ] Three-channel generation requirement understood.
[ ] z-score/inverse-transform policy understood.
[ ] Synthetic provenance requirements understood.

Remaining issue / comment:


PERSON 4
--------
Checker:
Date:
Status: OPEN-LIMITATION (pending recipient check)

Checks:
[ ] Dataset card reviewed.
[ ] Budget/access table reviewed.
[ ] Reproducibility evidence reviewed.
[ ] Open limitations reviewed.
[ ] Forbidden unsupported claims reviewed.

Remaining issue / comment:


FINAL RECIPIENT ACCEPTANCE
--------------------------
Do not mark PASS until the relevant recipients have actually
performed and recorded their checks.
"""

ACCEPTANCE_OUT.write_text(
    acceptance,
    encoding="utf-8",
)


# ============================================================
# HANDOFF CHECKLIST
# ============================================================

checklist = f"""EPSON PERSON 1 - HANDOFF CHECKLIST
========================================================================

Release:
EPSON_PERSON1_HANDOFF_V1

Generated:
{created}

Allowed statuses:
PASS / FAIL / OPEN-LIMITATION


01. SOURCE / LABEL / UNIT TRACEABILITY
Requirement:
Source, labels, channels and units are documented.

Evidence:
dataset_card.txt
source_register.csv
public_manifest.csv

Result:
PASS

Checker:
Person 1 automated/manual evidence review

Remaining issue:
Dataset-use / redistribution permission remains not fully verified
for every possible external/public/commercial use.


02. SIGNAL QA
Requirement:
Train signal quality was reviewed beyond shape/NaN checks.

Evidence:
signal_qc_train_k1.csv
channel_relations_train_k1.csv
signal_review.txt
signal_qa_protocol.json

Result:
OPEN-LIMITATION

Checker:
Person 1

Remaining issue:
One plausible transient retained; no timestamp-jitter analysis because
per-sample timestamps are unavailable.


03. CROSS-SPLIT INTEGRITY / NEAR DUPLICATE REVIEW
Requirement:
Exact, near and common-shift overlap risks are reviewed.

Evidence:
near_duplicate_stage3_final.json
near_duplicate_review.txt
near_duplicate_verification.csv

Result:
OPEN-LIMITATION

Checker:
Person 1

Remaining issue:
Confirmed overlap pairs = 0 under the frozen protocol, but this does
not prove complete session/source independence.


04. DATA CONTRACT / ACCESS POLICY
Requirement:
Split, scarcity pools, scaler scope and test policy are frozen.

Evidence:
data_contract.json
access_and_scarcity.csv
protocol_frozen.txt

Result:
PASS

Checker:
Person 1 automated validation

Remaining issue:
None for the frozen file-level protocol.


05. LOADER
Requirement:
Consumer can load frozen prepared tensors without reprocessing raw.

Evidence:
loader_v1/
loader_selftest.json
consumer_smoke_test.json

Result:
PASS

Checker:
Person 1 producer-side readiness test

Remaining issue:
Actual recipient still needs to run the supplied smoke test.


06. INDEPENDENT REPRODUCTION
Requirement:
New output directory reproduces arrays, IDs, scaler and metadata.

Evidence:
reproducibility_report.json
reproducibility_report.txt
stage6_reproduction_final.json

Result:
PASS

Checker:
Person 1 automated reproduction comparison

Remaining issue:
Same-machine/environment reproduction only; cross-machine reproduction
has not been demonstrated.


07. VERSION / PACKAGE TRACEABILITY
Requirement:
Environment and artifact checksums are recorded.

Evidence:
VERSION.txt
environment_reproducibility.txt
pip_freeze.txt
package_manifest.csv
stage6_final.json

Result:
OPEN-LIMITATION

Checker:
Person 1

Remaining issue:
No Git repository/commit history was available at release time;
SHA256 + frozen protocol are the version authority.


08. PERSON 2 ACCEPTANCE
Requirement:
Detector/augmentation consumer independently verifies correct use.

Evidence:
RECIPIENT_SMOKE_TEST.txt
RECIPIENT_ACCEPTANCE_TEMPLATE.txt

Result:
OPEN-LIMITATION

Checker:
PENDING - actual Person 2

Remaining issue:
Recipient has not yet recorded acceptance.


09. PERSON 3 ACCEPTANCE
Requirement:
Generator consumer independently verifies scarcity and synthetic-data policy.

Evidence:
RECIPIENT_SMOKE_TEST.txt
RECIPIENT_ACCEPTANCE_TEMPLATE.txt

Result:
OPEN-LIMITATION

Checker:
PENDING - actual Person 3

Remaining issue:
Recipient has not yet recorded acceptance.


10. PERSON 4 ACCEPTANCE
Requirement:
Presentation/report consumer reviews claims and limitations.

Evidence:
HANDOFF_README.txt
RECIPIENT_ACCEPTANCE_TEMPLATE.txt

Result:
OPEN-LIMITATION

Checker:
PENDING - actual Person 4

Remaining issue:
Recipient has not yet recorded acceptance.


OVERALL PRODUCER READINESS
--------------------------
PASS

OVERALL RECIPIENT ACCEPTANCE
----------------------------
OPEN-LIMITATION

CURRENT HANDOFF STATE
---------------------
READY_FOR_RECIPIENT_ACCEPTANCE_WITH_OPEN_LIMITATIONS

IMPORTANT
---------
This checklist does not claim zero leakage or session independence.
No recipient acceptance has been invented or signed by Person 1.
"""

CHECKLIST_OUT.write_text(
    checklist,
    encoding="utf-8",
)


# ============================================================
# HANDOFF MANIFEST
# ============================================================

manifest_paths = [
    DATASET_CARD,
    SOURCE_REGISTER,
    ACCESS,
    CONTRACT,
    QA / "protocol_frozen.txt",
    STAGE3,
    LOADER_TEST,
    STAGE6,
    REPRO_FINAL,
    REPRO_REPORT,
    PACKAGE_MANIFEST,
    SMOKE_OUT,
    README_OUT,
    SMOKE_GUIDE_OUT,
    ACCEPTANCE_OUT,
    CHECKLIST_OUT,
    Path(__file__),
]


manifest = {
    "release":
        "EPSON_PERSON1_HANDOFF_V1",

    "created_at_local":
        created,

    "purpose":
        "Minimal handoff evidence index",

    "files": [],
}


for p in manifest_paths:

    manifest[
        "files"
    ].append(
        {
            "path":
                str(
                    p.relative_to(
                        ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),

            "bytes":
                p.stat().st_size,

            "sha256":
                sha256_file(
                    p
                ),
        }
    )


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# STAGE 7 READINESS
# ============================================================

final = {
    "stage": 7,

    "name":
        "handoff_readiness_and_recipient_acceptance",

    "producer_readiness":
        "PASS",

    "recipient_acceptance":
        "OPEN-LIMITATION",

    "recipient_acceptance_complete":
        False,

    "status":
        "READY_FOR_RECIPIENT_ACCEPTANCE_WITH_OPEN_LIMITATIONS",

    "consumer_smoke_test":
        "PASS",

    "stage6_exact_reproduction":
        True,

    "handoff_manifest_sha256":
        sha256_file(
            MANIFEST_OUT
        ),

    "open_limitations": [
        "Actual Person 2 acceptance pending.",
        "Actual Person 3 acceptance pending.",
        "Actual Person 4 acceptance pending.",
        "Recording-session independence unverified.",
        "Per-sample timestamps unavailable.",
        "Single rig / nominal speed.",
        "Near-duplicate review cannot prove absolute source independence.",
        "Dataset-use/redistribution rights require scope-specific verification.",
        "No Git commit history at release time.",
        "Cross-machine reproduction not yet demonstrated.",
    ],

    "next_action":
        (
            "Provide the handoff bundle to Persons 2/3/4 and "
            "have the actual recipients run RECIPIENT_SMOKE_TEST.txt "
            "and complete RECIPIENT_ACCEPTANCE_TEMPLATE.txt."
        ),
}


FINAL_OUT.write_text(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 72)
print("STAGE 7 - HANDOFF READINESS")
print("=" * 72)

print()
print("Prior-stage gates: PASS")
print("Consumer loader smoke test: PASS")

print(
    "k1 X:",
    list(
        k1.X.shape
    ),
    k1.X.dtype,
)

print(
    "k1 y_class:",
    sorted(
        np.unique(
            k1.y_class
        ).tolist()
    ),
)

print(
    "k1 y_binary:",
    sorted(
        np.unique(
            k1.y_binary
        ).tolist()
    ),
)

print(
    "Recording counts:",
    record_counts,
)

print(
    "Budget nesting k1 < k2 < k4:",
    k1_ids < k2_ids < k4_ids,
)

print(
    "k1 scaler fit IDs:",
    len(
        scaler_ids
    ),
)

print(
    "Validation in k1 scaler:",
    bool(
        scaler_ids
        &
        validation_ids
    ),
)

print(
    "Test in k1 scaler:",
    bool(
        scaler_ids
        &
        test_ids
    ),
)

print(
    "Inverse roundtrip max abs error:",
    inverse_roundtrip_max_abs_error,
)

print()
print("========== CREATED ==========")

for p in [
    SMOKE_OUT,
    README_OUT,
    SMOKE_GUIDE_OUT,
    ACCEPTANCE_OUT,
    CHECKLIST_OUT,
    MANIFEST_OUT,
    FINAL_OUT,
]:

    print(
        p.name,
        "| bytes=",
        p.stat().st_size,
        "| sha256=",
        sha256_file(
            p
        ),
    )

print()
print(
    "PRODUCER READINESS: PASS"
)

print(
    "RECIPIENT ACCEPTANCE: OPEN-LIMITATION"
)

print(
    "STAGE 7 STATUS:",
    final[
        "status"
    ],
)

print()
print(
    "No recipient acceptance was self-signed."
)

print()
print(
    "========== STAGE 7 READINESS COMPLETE =========="
)
