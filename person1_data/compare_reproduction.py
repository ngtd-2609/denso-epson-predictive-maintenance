from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json

import numpy as np


ROOT = Path(r"D:\DENSO_FACTORY\person1_data")

REFERENCE = (
    ROOT
    / "results"
    / "prepared_v1"
)

REPRO = (
    ROOT
    / "results"
    / "repro_release_v2"
)

QA = (
    ROOT
    / "results"
    / "qa_v2"
)

JSON_OUT = (
    QA
    / "reproducibility_report.json"
)

TXT_OUT = (
    QA
    / "reproducibility_report.txt"
)

FINAL_OUT = (
    QA
    / "stage6_reproduction_final.json"
)

PACKAGE_MANIFEST = (
    QA
    / "package_manifest.csv"
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


def load_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv_rows(path):

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# REQUIRED FILES
# ============================================================

for p in [
    REFERENCE / "preparation_report.json",
    REFERENCE / "verification.json",
    REPRO / "preparation_report.json",
    REPRO / "verification.json",
    PACKAGE_MANIFEST,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


if JSON_OUT.exists():
    raise RuntimeError(
        f"STOP: output already exists: {JSON_OUT}"
    )

if TXT_OUT.exists():
    raise RuntimeError(
        f"STOP: output already exists: {TXT_OUT}"
    )

if FINAL_OUT.exists():
    raise RuntimeError(
        f"STOP: output already exists: {FINAL_OUT}"
    )


# ============================================================
# VERIFY CODE USED FOR REPRODUCTION MATCHES RELEASE PACKAGE
# ============================================================

with PACKAGE_MANIFEST.open(
    "r",
    newline="",
    encoding="utf-8",
) as f:

    package_rows = list(
        csv.DictReader(f)
    )


package_hash = {
    row["relative_path"].replace("\\", "/"):
        row["sha256"]
    for row in package_rows
}


code_checks = {}


for relative in [
    "data_pipeline.py",
    "verify_prepared.py",
]:

    current = sha256_file(
        ROOT / relative
    )

    expected = package_hash.get(
        relative
    )

    match = (
        expected is not None
        and
        current == expected
    )

    code_checks[
        relative
    ] = {
        "current_sha256":
            current,

        "release_manifest_sha256":
            expected,

        "matches_release_manifest":
            match,
    }


if not all(
    x[
        "matches_release_manifest"
    ]
    for x in code_checks.values()
):
    raise RuntimeError(
        "Current reproduction code differs from "
        "the Stage-6 packaged code."
    )


# ============================================================
# PREPARATION REPORT COMPARISON
# ============================================================

reference_report = load_json(
    REFERENCE
    / "preparation_report.json"
)

repro_report = load_json(
    REPRO
    / "preparation_report.json"
)


preparation_report_equal = (
    reference_report
    ==
    repro_report
)


# ============================================================
# VERIFICATION COMPARISON
# ============================================================

reference_verify = load_json(
    REFERENCE
    / "verification.json"
)

repro_verify = load_json(
    REPRO
    / "verification.json"
)


verification_checks_equal = (
    reference_verify.get(
        "checks"
    )
    ==
    repro_verify.get(
        "checks"
    )
)


repro_all_verification_pass = all(
    v.get(
        "passed"
    ) is True
    for v in repro_verify.get(
        "checks",
        {}
    ).values()
)


verification_code_hash_equal = (
    reference_verify.get(
        "code_sha256"
    )
    ==
    repro_verify.get(
        "code_sha256"
    )
)


# ============================================================
# PROFILE CONTENT COMPARISON
# ============================================================

profiles = [
    "normal_only",
    "k1",
    "k2",
    "k4",
]

splits = [
    "train",
    "validation",
    "test",
]

required_npz_keys = {
    "X",
    "y_class",
    "y_binary",
    "recording_id",
    "start_sample",
}


profile_results = {}

all_scalers_equal = True
all_windows_equal = True
all_arrays_equal = True


for profile in profiles:

    ref_dir = (
        REFERENCE
        / profile
    )

    new_dir = (
        REPRO
        / profile
    )

    # --------------------------------------------------------
    # Scaler comparison
    # --------------------------------------------------------

    ref_scaler = load_json(
        ref_dir
        / "scaler.json"
    )

    new_scaler = load_json(
        new_dir
        / "scaler.json"
    )

    scaler_equal = (
        ref_scaler
        ==
        new_scaler
    )

    all_scalers_equal = (
        all_scalers_equal
        and
        scaler_equal
    )

    # --------------------------------------------------------
    # Windows CSV semantic comparison
    # --------------------------------------------------------

    ref_windows = load_csv_rows(
        ref_dir
        / "windows.csv"
    )

    new_windows = load_csv_rows(
        new_dir
        / "windows.csv"
    )

    windows_semantic_equal = (
        ref_windows
        ==
        new_windows
    )

    windows_byte_equal = (
        sha256_file(
            ref_dir
            / "windows.csv"
        )
        ==
        sha256_file(
            new_dir
            / "windows.csv"
        )
    )

    all_windows_equal = (
        all_windows_equal
        and
        windows_semantic_equal
    )

    result = {
        "scaler_semantic_equal":
            scaler_equal,

        "scaler_fit_recording_ids_equal":
            (
                ref_scaler[
                    "fit_recording_ids"
                ]
                ==
                new_scaler[
                    "fit_recording_ids"
                ]
            ),

        "scaler_mean_equal":
            (
                ref_scaler[
                    "mean"
                ]
                ==
                new_scaler[
                    "mean"
                ]
            ),

        "scaler_scale_equal":
            (
                ref_scaler[
                    "scale"
                ]
                ==
                new_scaler[
                    "scale"
                ]
            ),

        "windows_semantic_equal":
            windows_semantic_equal,

        "windows_byte_equal":
            windows_byte_equal,

        "window_rows":
            len(
                ref_windows
            ),

        "splits": {},
    }

    # --------------------------------------------------------
    # NPZ content comparison
    # --------------------------------------------------------

    for split in splits:

        ref_npz = (
            ref_dir
            / f"{split}.npz"
        )

        new_npz = (
            new_dir
            / f"{split}.npz"
        )

        with np.load(
            ref_npz,
            allow_pickle=False,
        ) as a, np.load(
            new_npz,
            allow_pickle=False,
        ) as b:

            keys_a = set(
                a.files
            )

            keys_b = set(
                b.files
            )

            keys_equal = (
                keys_a
                ==
                keys_b
                ==
                required_npz_keys
            )

            array_result = {
                "keys_equal":
                    keys_equal,

                "reference_npz_sha256":
                    sha256_file(
                        ref_npz
                    ),

                "reproduced_npz_sha256":
                    sha256_file(
                        new_npz
                    ),

                "npz_byte_hash_equal":
                    (
                        sha256_file(
                            ref_npz
                        )
                        ==
                        sha256_file(
                            new_npz
                        )
                    ),
            }

            split_exact = (
                keys_equal
            )

            for key in sorted(
                required_npz_keys
            ):

                arr_a = a[
                    key
                ]

                arr_b = b[
                    key
                ]

                shape_equal = (
                    arr_a.shape
                    ==
                    arr_b.shape
                )

                dtype_equal = (
                    arr_a.dtype
                    ==
                    arr_b.dtype
                )

                exact_equal = (
                    shape_equal
                    and
                    dtype_equal
                    and
                    np.array_equal(
                        arr_a,
                        arr_b,
                    )
                )

                entry = {
                    "shape_reference":
                        list(
                            arr_a.shape
                        ),

                    "shape_reproduced":
                        list(
                            arr_b.shape
                        ),

                    "dtype_reference":
                        str(
                            arr_a.dtype
                        ),

                    "dtype_reproduced":
                        str(
                            arr_b.dtype
                        ),

                    "exact_equal":
                        bool(
                            exact_equal
                        ),
                }

                if (
                    key == "X"
                    and
                    shape_equal
                ):

                    diff = np.abs(
                        arr_a.astype(
                            np.float64
                        )
                        -
                        arr_b.astype(
                            np.float64
                        )
                    )

                    entry[
                        "max_abs_difference"
                    ] = float(
                        diff.max()
                    )

                    entry[
                        "mean_abs_difference"
                    ] = float(
                        diff.mean()
                    )

                array_result[
                    key
                ] = entry

                split_exact = (
                    split_exact
                    and
                    exact_equal
                )

            array_result[
                "all_content_exact"
            ] = bool(
                split_exact
            )

            all_arrays_equal = (
                all_arrays_equal
                and
                split_exact
            )

            result[
                "splits"
            ][
                split
            ] = array_result

    profile_results[
        profile
    ] = result


# ============================================================
# FINAL DECISION
# ============================================================

overall_pass = all(
    [
        preparation_report_equal,
        verification_checks_equal,
        repro_all_verification_pass,
        verification_code_hash_equal,
        all_scalers_equal,
        all_windows_equal,
        all_arrays_equal,
        all(
            x[
                "matches_release_manifest"
            ]
            for x in code_checks.values()
        ),
    ]
)


status = (
    "PASS"
    if overall_pass
    else
    "FAIL"
)


created = (
    datetime.now()
    .astimezone()
    .isoformat()
)


report = {
    "stage": 6,
    "test":
        "independent_prepared_data_reproduction",

    "created_at_local":
        created,

    "status":
        status,

    "reference":
        "results/prepared_v1",

    "reproduced":
        "results/repro_release_v2",

    "same_output_directory":
        False,

    "same_machine_environment":
        True,

    "code_matches_stage6_package_manifest":
        all(
            x[
                "matches_release_manifest"
            ]
            for x in code_checks.values()
        ),

    "code_checks":
        code_checks,

    "preparation_report_semantic_equal":
        preparation_report_equal,

    "verification_checks_equal":
        verification_checks_equal,

    "verification_code_hash_equal":
        verification_code_hash_equal,

    "reproduced_verification_all_pass":
        repro_all_verification_pass,

    "all_scalers_semantically_equal":
        all_scalers_equal,

    "all_windows_metadata_semantically_equal":
        all_windows_equal,

    "all_npz_array_content_exact":
        all_arrays_equal,

    "profiles":
        profile_results,

    "interpretation": (
        "Independent materialization into a new output directory "
        "reproduced the frozen arrays, labels, recording IDs, "
        "window starts, scalers and window metadata exactly."
        if overall_pass
        else
        "At least one reproduced artifact differs from the "
        "frozen prepared_v1 reference; Stage 6 must remain open."
    ),

    "limitation": (
        "This is an independent output-directory rerun on the "
        "same machine/environment, not a cross-machine reproduction."
    ),
}


JSON_OUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

lines = []

lines.append(
    "EPSON STAGE 6 - INDEPENDENT REPRODUCTION REPORT"
)

lines.append(
    "=" * 72
)

lines.append(
    f"Status: {status}"
)

lines.append(
    f"Reference: {REFERENCE}"
)

lines.append(
    f"Reproduced: {REPRO}"
)

lines.append("")

lines.append(
    "Code matches packaged release: "
    + str(
        report[
            "code_matches_stage6_package_manifest"
        ]
    )
)

lines.append(
    "Preparation report equal: "
    + str(
        preparation_report_equal
    )
)

lines.append(
    "Verification checks equal: "
    + str(
        verification_checks_equal
    )
)

lines.append(
    "All reproduced verification checks PASS: "
    + str(
        repro_all_verification_pass
    )
)

lines.append(
    "All scalers equal: "
    + str(
        all_scalers_equal
    )
)

lines.append(
    "All windows metadata equal: "
    + str(
        all_windows_equal
    )
)

lines.append(
    "All NPZ array contents exact: "
    + str(
        all_arrays_equal
    )
)

lines.append("")


for profile in profiles:

    p = profile_results[
        profile
    ]

    lines.append(
        f"[{profile}]"
    )

    lines.append(
        " scaler equal: "
        + str(
            p[
                "scaler_semantic_equal"
            ]
        )
    )

    lines.append(
        " windows metadata equal: "
        + str(
            p[
                "windows_semantic_equal"
            ]
        )
    )

    for split in splits:

        s = p[
            "splits"
        ][
            split
        ]

        lines.append(
            f" {split}: "
            f"content_exact={s['all_content_exact']}, "
            f"X_shape={s['X']['shape_reference']}, "
            f"X_max_abs_diff={s['X'].get('max_abs_difference')}, "
            f"npz_byte_hash_equal={s['npz_byte_hash_equal']}"
        )

    lines.append("")


lines.append(
    "Important:"
)

lines.append(
    "- NPZ byte hashes are recorded but are NOT the primary "
    "reproducibility criterion."
)

lines.append(
    "- PASS requires exact array/ID/start/scaler/window-metadata "
    "agreement."
)

lines.append(
    "- This rerun used a new output directory on the same "
    "machine/environment."
)


TXT_OUT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


final = {
    "stage": 6,
    "name":
        "independent_reproduction_closure",

    "status":
        (
            "PASS_WITH_OPEN_LIMITATION"
            if overall_pass
            else
            "FAIL"
        ),

    "independent_output_directory_reproduction":
        bool(
            overall_pass
        ),

    "array_content_exact":
        bool(
            all_arrays_equal
        ),

    "scalers_exact":
        bool(
            all_scalers_equal
        ),

    "window_metadata_exact":
        bool(
            all_windows_equal
        ),

    "cross_machine_reproduction":
        False,

    "open_limitations": [
        "No Git repository was available at release time.",
        "Reproduction was performed on the same machine/environment.",
        "Recording-session independence remains unverified.",
        "Dataset represents a single rig and nominal speed.",
    ],

    "next_stage_allowed":
        bool(
            overall_pass
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
print("REPRODUCTION CONTENT COMPARISON")
print("=" * 72)

print()
print(
    "Code matches packaged release:",
    report[
        "code_matches_stage6_package_manifest"
    ],
)

print(
    "Preparation report equal:",
    preparation_report_equal,
)

print(
    "Verification checks equal:",
    verification_checks_equal,
)

print(
    "Reproduced verification all PASS:",
    repro_all_verification_pass,
)

print(
    "All scalers equal:",
    all_scalers_equal,
)

print(
    "All windows metadata equal:",
    all_windows_equal,
)

print(
    "All NPZ array contents exact:",
    all_arrays_equal,
)


for profile in profiles:

    print()
    print(
        f"========== {profile} =========="
    )

    print(
        "scaler equal:",
        profile_results[
            profile
        ][
            "scaler_semantic_equal"
        ],
    )

    print(
        "windows equal:",
        profile_results[
            profile
        ][
            "windows_semantic_equal"
        ],
    )

    for split in splits:

        result = profile_results[
            profile
        ][
            "splits"
        ][
            split
        ]

        print(
            f"{split:10s}",
            "| exact=",
            result[
                "all_content_exact"
            ],
            "| X shape=",
            result[
                "X"
            ][
                "shape_reference"
            ],
            "| X max abs diff=",
            result[
                "X"
            ].get(
                "max_abs_difference"
            ),
            "| NPZ byte hash equal=",
            result[
                "npz_byte_hash_equal"
            ],
        )


print()
print(
    "REPRODUCTION STATUS:",
    status
)

print(
    "NEXT STAGE ALLOWED:",
    final[
        "next_stage_allowed"
    ]
)

print()
print("Created:")
print(JSON_OUT)
print(TXT_OUT)
print(FINAL_OUT)

print()
print(
    "========== REPRODUCTION COMPARISON COMPLETE =========="
)


if not overall_pass:
    raise SystemExit(2)
