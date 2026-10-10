from pathlib import Path
import hashlib
import json
import sys

import numpy as np


ROOT = Path(
    r"D:\DENSO_FACTORY\person1_data"
)

sys.path.insert(
    0,
    str(ROOT)
)

from loader_v1 import EpsonPreparedLoader


QA = (
    ROOT
    / "results"
    / "qa_v2"
)

OUT = (
    QA
    / "loader_selftest.json"
)


def digest(path):
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


def main():

    if OUT.exists():
        raise RuntimeError(
            f"STOP: self-test output already exists: {OUT}"
        )

    loader = EpsonPreparedLoader(
        ROOT
    )

    print("=" * 72)
    print("EPSON LOADER V1 SELF-TEST")
    print("=" * 72)

    print(
        "Prepared status:",
        loader.report["status"],
    )

    print(
        "Contract status:",
        loader.contract["status"],
    )

    results = {}

    for profile in [
        "normal_only",
        "k1",
        "k2",
        "k4",
    ]:

        print()
        print(
            f"========== {profile} =========="
        )

        scaler = loader.get_scaler(
            profile
        )

        print(
            "Scaler fit records:",
            len(
                scaler[
                    "fit_recording_ids"
                ]
            ),
        )

        results[
            profile
        ] = {}

        for split in [
            "train",
            "validation",
            "test",
        ]:

            data = loader.load(
                profile,
                split,
            )

            unique_records = len(
                set(
                    data.recording_id.tolist()
                )
            )

            classes = sorted(
                np.unique(
                    data.y_class
                ).tolist()
            )

            print(
                f"{split:10s}",
                "| X=",
                list(
                    data.X.shape
                ),
                "| records=",
                unique_records,
                "| classes=",
                classes,
            )

            results[
                profile
            ][
                split
            ] = {
                "X_shape":
                    list(
                        data.X.shape
                    ),

                "X_dtype":
                    str(
                        data.X.dtype
                    ),

                "unique_recordings":
                    unique_records,

                "classes":
                    classes,

                "passed":
                    True,
            }

    # --------------------------------------------------------
    # Alias check
    # --------------------------------------------------------

    alias = loader.load(
        "1",
        "val",
    )

    canonical = loader.load(
        "k1",
        "validation",
    )

    if alias.X.shape != canonical.X.shape:
        raise RuntimeError(
            "Alias resolution failed"
        )

    print()
    print(
        "Alias check: PASS"
    )

    # --------------------------------------------------------
    # PyTorch adapter smoke test.
    # Optional dependency; absence is not loader failure.
    # --------------------------------------------------------

    torch_status = (
        "NOT_INSTALLED_OPTIONAL"
    )

    try:
        train = loader.load(
            "k1",
            "train",
        )

        ds = loader.to_torch_dataset(
            train,
            target="class",
        )

        item = ds[0]

        if tuple(
            item[
                "X"
            ].shape
        ) != (
            3,
            3000,
        ):
            raise RuntimeError(
                "Torch dataset X shape incorrect"
            )

        torch_status = "PASS"

        print(
            "PyTorch adapter: PASS"
        )

    except ImportError:

        print(
            "PyTorch adapter: NOT INSTALLED "
            "(optional, NumPy loader still PASS)"
        )

    loader_file = (
        ROOT
        / "loader_v1"
        / "epson_loader.py"
    )

    init_file = (
        ROOT
        / "loader_v1"
        / "__init__.py"
    )

    final = {
        "stage": 5,
        "package": "loader_v1",

        "status": "PASS",

        "prepared_source":
            "results/prepared_v1",

        "window_layout":
            "[N, 3, 3000]",

        "profiles": [
            "normal_only",
            "k1",
            "k2",
            "k4",
        ],

        "splits": [
            "train",
            "validation",
            "test",
        ],

        "results":
            results,

        "alias_check":
            "PASS",

        "pytorch_adapter":
            torch_status,

        "loader_sha256":
            digest(
                loader_file
            ),

        "init_sha256":
            digest(
                init_file
            ),

        "important_rule":
            (
                "Loader reads frozen prepared_v1 only; "
                "it does not refit scaler or re-window raw data."
            ),
    }

    OUT.write_text(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("LOADER SELF-TEST STATUS: PASS")
    print("=" * 72)

    print()
    print(
        "Saved:",
        OUT
    )


if __name__ == "__main__":
    main()
