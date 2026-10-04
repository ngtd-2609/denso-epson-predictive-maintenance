from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import platform
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy

try:
    import torch
    TORCH_VERSION = torch.__version__
except Exception:
    TORCH_VERSION = "NOT_AVAILABLE"


ROOT = Path(r"D:\DENSO_FACTORY\person1_data")
RESULTS = ROOT / "results"
QA = RESULTS / "qa_v2"

VERSION_OUT = ROOT / "VERSION.txt"

ENV_OUT = QA / "environment_reproducibility.txt"
PIP_OUT = QA / "pip_freeze.txt"
PACKAGE_OUT = QA / "package_manifest.csv"
REPRO_OUT = QA / "reproducibility_manifest.json"
FINAL_OUT = QA / "stage6_final.json"

CONTRACT = QA / "data_contract.json"
STAGE3 = QA / "near_duplicate_stage3_final.json"
LOADER_TEST = QA / "loader_selftest.json"

PREP_REPORT = RESULTS / "prepared_v1" / "preparation_report.json"
PREP_VERIFY = RESULTS / "prepared_v1" / "verification.json"


# ------------------------------------------------------------
# DO NOT OVERWRITE RELEASE ARTIFACTS
# ------------------------------------------------------------

for p in [
    VERSION_OUT,
    ENV_OUT,
    PIP_OUT,
    PACKAGE_OUT,
    REPRO_OUT,
    FINAL_OUT,
]:
    if p.exists():
        raise RuntimeError(
            f"STOP: Stage-6 artifact already exists: {p}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def rel(path):
    return str(
        path.relative_to(ROOT)
    ).replace("\\", "/")


def git_info():

    command = [
        "git",
        "-C",
        str(ROOT),
        "rev-parse",
        "--show-toplevel",
    ]

    p = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if p.returncode != 0:

        return {
            "available": False,
            "repository_root": None,
            "branch": None,
            "commit": None,
            "working_tree_clean": None,
            "note": (
                "person1_data is not inside a Git repository "
                "at release time."
            ),
        }

    repo = p.stdout.strip()

    branch = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "branch",
            "--show-current",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    commit = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    status = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    return {
        "available": True,
        "repository_root": repo,
        "branch": branch,
        "commit": commit,
        "working_tree_clean": status == "",
        "note": None,
    }


# ------------------------------------------------------------
# REQUIRED GATES
# ------------------------------------------------------------

required = [
    CONTRACT,
    STAGE3,
    LOADER_TEST,
    PREP_REPORT,
    PREP_VERIFY,
]

missing = [
    str(p)
    for p in required
    if not p.exists()
]

if missing:
    raise RuntimeError(
        "Missing required release inputs:\n"
        + "\n".join(missing)
    )


contract = read_json(CONTRACT)
stage3 = read_json(STAGE3)
loader = read_json(LOADER_TEST)
prep = read_json(PREP_REPORT)
verify = read_json(PREP_VERIFY)


if contract.get("status") != "FROZEN":
    raise RuntimeError(
        "Data contract is not FROZEN"
    )

if stage3.get(
    "next_stage_allowed"
) is not True:
    raise RuntimeError(
        "Stage 3 gate not satisfied"
    )

if stage3.get(
    "confirmed_overlap_pairs"
) != 0:
    raise RuntimeError(
        "Confirmed overlap exists"
    )

if loader.get("status") != "PASS":
    raise RuntimeError(
        "Loader self-test is not PASS"
    )

if prep.get("status") != "complete":
    raise RuntimeError(
        "Prepared data is not complete"
    )


checks = verify.get(
    "checks",
    {}
)

expected_profiles = {
    "normal_only",
    "k1",
    "k2",
    "k4",
}

if set(checks) != expected_profiles:
    raise RuntimeError(
        "Prepared verification does not cover all profiles"
    )

if not all(
    checks[p].get("passed") is True
    for p in expected_profiles
):
    raise RuntimeError(
        "A prepared-data verification check failed"
    )


# ------------------------------------------------------------
# RELEASE VERSION
# ------------------------------------------------------------

created = (
    datetime.now()
    .astimezone()
    .isoformat()
)

release_name = "EPSON_PERSON1_HANDOFF_V1"

git = git_info()


version_text = f"""EPSON PERSON 1 RELEASE
======================

Release:
{release_name}

Created:
{created}

Dataset:
Epson trimmed

Protocol:
data_contract_v1 / FROZEN

Prepared data:
prepared_v1 / complete

Loader:
loader_v1 / PASS

Git available:
{git["available"]}

Version authority:
Frozen protocol + SHA256 artifact manifest.

Important:
Do not silently replace split, prepared data, scaler,
scarcity membership or protocol artifacts under this release.
"""

VERSION_OUT.write_text(
    version_text,
    encoding="utf-8",
)


# ------------------------------------------------------------
# ENVIRONMENT SNAPSHOT
# ------------------------------------------------------------

environment = f"""EPSON PERSON 1 - REPRODUCIBILITY ENVIRONMENT
================================================

Release:
{release_name}

Captured:
{created}

Python executable:
{sys.executable}

Python:
{platform.python_version()}

Platform:
{platform.platform()}

numpy:
{np.__version__}

pandas:
{pd.__version__}

scipy:
{scipy.__version__}

matplotlib:
{matplotlib.__version__}

torch:
{TORCH_VERSION}

Git available:
{git["available"]}

Git repository:
{git["repository_root"]}

Git branch:
{git["branch"]}

Git commit:
{git["commit"]}

Git working tree clean:
{git["working_tree_clean"]}

Git note:
{git["note"]}

Reproducibility rule:
Use the frozen prepared_v1 artifacts and loader_v1.
Do not refit the scaler or regenerate scarcity membership
unless explicitly creating a new protocol version.
"""

ENV_OUT.write_text(
    environment,
    encoding="utf-8",
)


# ------------------------------------------------------------
# FULL PIP SNAPSHOT
# ------------------------------------------------------------

pip = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "freeze",
    ],
    capture_output=True,
    text=True,
)

if pip.returncode != 0:
    raise RuntimeError(
        "pip freeze failed:\n"
        + pip.stderr
    )

PIP_OUT.write_text(
    pip.stdout,
    encoding="utf-8",
)


# ------------------------------------------------------------
# PACKAGE MANIFEST
# ------------------------------------------------------------

excluded_stage6 = {
    PACKAGE_OUT.resolve(),
    REPRO_OUT.resolve(),
    FINAL_OUT.resolve(),
}


files = []


def add_files(category, paths, required, note):

    for p in paths:

        p = Path(p)

        if not p.is_file():
            continue

        if p.resolve() in excluded_stage6:
            continue

        files.append(
            {
                "category": category,
                "relative_path": rel(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "required_for_handoff": required,
                "note": note,
            }
        )


# Main code
add_files(
    "code",
    sorted(
        ROOT.glob("*.py")
    ),
    "YES",
    "Person-1 processing/QA/release code.",
)

# Loader package
add_files(
    "loader",
    sorted(
        (
            ROOT
            / "loader_v1"
        ).glob("*")
    ),
    "YES",
    "Frozen prepared-data loader package.",
)

# Audit and split
add_files(
    "audit",
    sorted(
        (
            RESULTS
            / "audit_v1"
        ).rglob("*")
    ),
    "YES",
    "Dataset audit/provenance artifacts.",
)

add_files(
    "split",
    sorted(
        (
            RESULTS
            / "split_seed42"
        ).rglob("*")
    ),
    "YES",
    "Frozen recording-level split artifacts.",
)

# Prepared data, including NPZ
add_files(
    "prepared",
    sorted(
        (
            RESULTS
            / "prepared_v1"
        ).rglob("*")
    ),
    "YES",
    "Frozen model-consumable prepared data.",
)

# QA Stage 1-5 supporting evidence
add_files(
    "qa",
    sorted(
        QA.rglob("*")
    ),
    "YES",
    "QA and protocol evidence.",
)

# Release/environment files outside QA scan
add_files(
    "release",
    [
        VERSION_OUT,
        ENV_OUT,
        PIP_OUT,
    ],
    "YES",
    "Release version and environment snapshot.",
)


# De-duplicate paths
dedup = {}

for row in files:
    dedup[
        row["relative_path"]
    ] = row

files = [
    dedup[k]
    for k in sorted(dedup)
]


with PACKAGE_OUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "relative_path",
            "bytes",
            "sha256",
            "required_for_handoff",
            "note",
        ],
    )

    writer.writeheader()
    writer.writerows(files)


# ------------------------------------------------------------
# REPRODUCIBILITY MANIFEST
# ------------------------------------------------------------

total_bytes = sum(
    int(r["bytes"])
    for r in files
)

categories = {}

for r in files:
    categories[
        r["category"]
    ] = (
        categories.get(
            r["category"],
            0,
        )
        + 1
    )


repro = {
    "release": release_name,
    "created_at_local": created,
    "status": (
        "PASS_WITH_OPEN_LIMITATION"
        if not git["available"]
        else
        "PASS"
    ),

    "versioning": {
        "release_label": release_name,
        "git": git,
        "fallback_version_authority": (
            "SHA256 package manifest and frozen protocol"
        ),
    },

    "environment": {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": TORCH_VERSION,
    },

    "protocol": {
        "data_contract_status":
            contract["status"],

        "contract_version":
            contract["contract_version"],

        "prepared_status":
            prep["status"],

        "stage3_status":
            stage3["status"],

        "confirmed_overlap_pairs":
            stage3[
                "confirmed_overlap_pairs"
            ],

        "loader_selftest_status":
            loader["status"],
    },

    "package": {
        "manifest":
            rel(PACKAGE_OUT),

        "manifest_sha256":
            sha256_file(
                PACKAGE_OUT
            ),

        "file_count":
            len(files),

        "total_bytes":
            total_bytes,

        "category_counts":
            categories,
    },

    "source_data_policy": {
        "raw_data_copied_into_release_manifest":
            False,

        "reason": (
            "Raw source files are not duplicated here; "
            "their audit checksums are already recorded in "
            "audit/preparation artifacts."
        ),

        "prepared_artifacts_hashed":
            True,
    },

    "open_limitations": [
        (
            "No Git repository/commit was available at release time."
            if not git["available"]
            else
            None
        ),
        "Recording-session independence remains unverified.",
        "Dataset is a single rig / nominal speed.",
    ],

    "next_stage_allowed": True,
}

repro[
    "open_limitations"
] = [
    x
    for x in repro[
        "open_limitations"
    ]
    if x is not None
]


REPRO_OUT.write_text(
    json.dumps(
        repro,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# FINAL STAGE-6 RECORD
# ------------------------------------------------------------

stage6 = {
    "stage": 6,
    "name": "reproducibility_version_package",
    "release": release_name,

    "status": repro["status"],

    "git_available":
        git["available"],

    "package_file_count":
        len(files),

    "package_total_bytes":
        total_bytes,

    "artifacts": {
        "VERSION.txt":
            sha256_file(
                VERSION_OUT
            ),

        "environment_reproducibility.txt":
            sha256_file(
                ENV_OUT
            ),

        "pip_freeze.txt":
            sha256_file(
                PIP_OUT
            ),

        "package_manifest.csv":
            sha256_file(
                PACKAGE_OUT
            ),

        "reproducibility_manifest.json":
            sha256_file(
                REPRO_OUT
            ),
    },

    "open_limitations":
        repro[
            "open_limitations"
        ],

    "next_stage_allowed":
        True,
}


FINAL_OUT.write_text(
    json.dumps(
        stage6,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("=" * 72)
print("STAGE 6 - REPRODUCIBILITY / VERSION / PACKAGE")
print("=" * 72)

print()
print("Release:", release_name)
print("Protocol:", contract["status"])
print("Prepared:", prep["status"])
print("Loader:", loader["status"])

print()
print("Git available:", git["available"])

if not git["available"]:
    print(
        "Git limitation recorded: "
        "release is versioned by frozen protocol + SHA256."
    )

print()
print("Environment:")
print(" Python     :", platform.python_version())
print(" numpy      :", np.__version__)
print(" pandas     :", pd.__version__)
print(" scipy      :", scipy.__version__)
print(" matplotlib :", matplotlib.__version__)
print(" torch      :", TORCH_VERSION)

print()
print("Package file count:", len(files))
print("Package bytes:", total_bytes)

print()
print("Category counts:")

for key in sorted(categories):
    print(
        f" {key:10s}:",
        categories[key]
    )

print()
print("========== CREATED ==========")

for p in [
    VERSION_OUT,
    ENV_OUT,
    PIP_OUT,
    PACKAGE_OUT,
    REPRO_OUT,
    FINAL_OUT,
]:
    print(
        p,
        "| bytes=",
        p.stat().st_size,
        "| sha256=",
        sha256_file(p),
    )

print()
print("STAGE 6 STATUS:", stage6["status"])
print("NEXT STAGE ALLOWED:", stage6["next_stage_allowed"])

print()
print("========== STAGE 6 COMPLETE ==========")
