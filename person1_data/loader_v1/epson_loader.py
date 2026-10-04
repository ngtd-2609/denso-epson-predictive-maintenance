"""
Frozen Epson prepared-data loader.

Important:
- Reads prepared_v1 only.
- Does NOT refit a scaler.
- Does NOT re-window raw data.
- Does NOT modify train/validation/test membership.
- Does NOT perform model training or selection.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


VALID_PROFILES = (
    "normal_only",
    "k1",
    "k2",
    "k4",
)

VALID_SPLITS = (
    "train",
    "validation",
    "test",
)


@dataclass(frozen=True)
class PreparedSplit:
    X: np.ndarray
    y_class: np.ndarray
    y_binary: np.ndarray
    recording_id: np.ndarray
    start_sample: np.ndarray
    profile: str
    split: str

    def __len__(self):
        return int(self.X.shape[0])

    @property
    def shape(self):
        return self.X.shape


class EpsonPreparedLoader:

    def __init__(self, root=None):

        if root is None:
            root = (
                Path(__file__)
                .resolve()
                .parents[1]
            )

        self.root = Path(root)

        self.prepared_dir = (
            self.root
            / "results"
            / "prepared_v1"
        )

        self.qa_dir = (
            self.root
            / "results"
            / "qa_v2"
        )

        self.report_path = (
            self.prepared_dir
            / "preparation_report.json"
        )

        self.verification_path = (
            self.prepared_dir
            / "verification.json"
        )

        self.contract_path = (
            self.qa_dir
            / "data_contract.json"
        )

        self.protocol_path = (
            self.qa_dir
            / "protocol_frozen.txt"
        )

        self.split_path = (
            self.root
            / "results"
            / "split_seed42"
            / "split.csv"
        )

        required = [
            self.report_path,
            self.verification_path,
            self.contract_path,
            self.protocol_path,
            self.split_path,
        ]

        missing = [
            str(p)
            for p in required
            if not p.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required frozen artifacts missing:\n"
                + "\n".join(missing)
            )

        self.report = json.loads(
            self.report_path.read_text(
                encoding="utf-8"
            )
        )

        self.verification = json.loads(
            self.verification_path.read_text(
                encoding="utf-8"
            )
        )

        self.contract = json.loads(
            self.contract_path.read_text(
                encoding="utf-8"
            )
        )

        self.split_table = pd.read_csv(
            self.split_path
        )

        self._validate_global_contract()


    def _validate_global_contract(self):

        if self.report.get("status") != "complete":
            raise RuntimeError(
                "prepared_v1 is not marked complete"
            )

        if self.contract.get("status") != "FROZEN":
            raise RuntimeError(
                "data contract is not FROZEN"
            )

        if (
            self.report.get("length") != 3000
            or
            self.report.get("stride") != 3000
        ):
            raise RuntimeError(
                "Unexpected frozen window specification"
            )

        if self.report.get(
            "channels"
        ) != ["X", "Y", "Z"]:
            raise RuntimeError(
                "Unexpected channel order"
            )

        if self.report.get(
            "fs_hz"
        ) != 3000:
            raise RuntimeError(
                "Unexpected sampling rate"
            )

        checks = self.verification.get(
            "checks",
            {}
        )

        required_checks = {
            "normal_only",
            "k1",
            "k2",
            "k4",
        }

        if set(checks) != required_checks:
            raise RuntimeError(
                "verification.json does not cover "
                "all frozen profiles"
            )

        for name in required_checks:

            if checks[
                name
            ].get(
                "passed"
            ) is not True:
                raise RuntimeError(
                    f"Prepared verification failed: {name}"
                )

        split_counts = (
            self.split_table
            .groupby("split")
            .size()
            .to_dict()
        )

        if split_counts != {
            "train": 48,
            "validation": 24,
            "test": 24,
        }:
            raise RuntimeError(
                f"Unexpected split counts: {split_counts}"
            )


    @staticmethod
    def _normalize_profile(profile):

        profile = str(
            profile
        ).lower()

        aliases = {
            "1": "k1",
            "2": "k2",
            "4": "k4",
        }

        profile = aliases.get(
            profile,
            profile
        )

        if profile not in VALID_PROFILES:
            raise ValueError(
                "profile must be one of "
                f"{VALID_PROFILES}"
            )

        return profile


    @staticmethod
    def _normalize_split(split):

        split = str(
            split
        ).lower()

        aliases = {
            "val": "validation",
            "valid": "validation",
        }

        split = aliases.get(
            split,
            split
        )

        if split not in VALID_SPLITS:
            raise ValueError(
                "split must be one of "
                f"{VALID_SPLITS}"
            )

        return split


    def _profile_detail(self, profile):

        if profile == "normal_only":
            return self.report[
                "normal_only"
            ]

        k = profile[
            1:
        ]

        return self.report[
            "budgets"
        ][
            k
        ]


    def get_scaler(self, profile):

        profile = self._normalize_profile(
            profile
        )

        detail = self._profile_detail(
            profile
        )

        scaler_path = (
            self.prepared_dir
            / profile
            / "scaler.json"
        )

        if not scaler_path.exists():
            raise FileNotFoundError(
                scaler_path
            )

        disk_scaler = json.loads(
            scaler_path.read_text(
                encoding="utf-8"
            )
        )

        expected = detail[
            "scaler"
        ]

        if (
            disk_scaler[
                "fit_recording_ids"
            ]
            !=
            expected[
                "fit_recording_ids"
            ]
        ):
            raise RuntimeError(
                f"{profile}: scaler fit IDs changed"
            )

        np.testing.assert_allclose(
            np.asarray(
                disk_scaler["mean"],
                dtype=np.float64,
            ),
            np.asarray(
                expected["mean"],
                dtype=np.float64,
            ),
            rtol=0,
            atol=0,
        )

        np.testing.assert_allclose(
            np.asarray(
                disk_scaler["scale"],
                dtype=np.float64,
            ),
            np.asarray(
                expected["scale"],
                dtype=np.float64,
            ),
            rtol=0,
            atol=0,
        )

        return disk_scaler


    def load(self, profile, split):

        profile = self._normalize_profile(
            profile
        )

        split = self._normalize_split(
            split
        )

        detail = self._profile_detail(
            profile
        )

        npz_path = (
            self.prepared_dir
            / profile
            / f"{split}.npz"
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                npz_path
            )

        required_keys = {
            "X",
            "y_class",
            "y_binary",
            "recording_id",
            "start_sample",
        }

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:

            if set(
                z.files
            ) != required_keys:
                raise RuntimeError(
                    f"{npz_path.name}: unexpected keys "
                    f"{z.files}"
                )

            X = z[
                "X"
            ].copy()

            y_class = z[
                "y_class"
            ].copy()

            y_binary = z[
                "y_binary"
            ].copy()

            recording_id = z[
                "recording_id"
            ].astype(
                str
            ).copy()

            start_sample = z[
                "start_sample"
            ].copy()

        self._validate_loaded(
            profile=profile,
            split=split,
            detail=detail,
            X=X,
            y_class=y_class,
            y_binary=y_binary,
            recording_id=recording_id,
            start_sample=start_sample,
        )

        return PreparedSplit(
            X=X,
            y_class=y_class,
            y_binary=y_binary,
            recording_id=recording_id,
            start_sample=start_sample,
            profile=profile,
            split=split,
        )


    def _validate_loaded(
        self,
        profile,
        split,
        detail,
        X,
        y_class,
        y_binary,
        recording_id,
        start_sample,
    ):

        n = len(X)

        if X.dtype != np.float32:
            raise RuntimeError(
                f"{profile}/{split}: X dtype "
                f"is {X.dtype}, expected float32"
            )

        if X.ndim != 3:
            raise RuntimeError(
                f"{profile}/{split}: X must be 3D"
            )

        if X.shape[
            1:
        ] != (
            3,
            3000,
        ):
            raise RuntimeError(
                f"{profile}/{split}: unexpected X shape "
                f"{X.shape}"
            )

        if not np.isfinite(
            X
        ).all():
            raise RuntimeError(
                f"{profile}/{split}: nonfinite X"
            )

        if not (
            len(y_class)
            ==
            len(y_binary)
            ==
            len(recording_id)
            ==
            len(start_sample)
            ==
            n
        ):
            raise RuntimeError(
                f"{profile}/{split}: array length mismatch"
            )

        if y_class.dtype != np.int64:
            raise RuntimeError(
                f"{profile}/{split}: y_class must be int64"
            )

        if y_binary.dtype != np.int64:
            raise RuntimeError(
                f"{profile}/{split}: y_binary must be int64"
            )

        if start_sample.dtype != np.int64:
            raise RuntimeError(
                f"{profile}/{split}: start_sample must be int64"
            )

        if not np.array_equal(
            y_binary,
            (
                y_class != 0
            ).astype(
                np.int64
            ),
        ):
            raise RuntimeError(
                f"{profile}/{split}: binary labels inconsistent"
            )

        if not np.all(
            start_sample % 3000 == 0
        ):
            raise RuntimeError(
                f"{profile}/{split}: invalid window start"
            )

        expected_count = int(
            detail[
                "window_counts"
            ][
                split
            ]
        )

        if n != expected_count:
            raise RuntimeError(
                f"{profile}/{split}: expected "
                f"{expected_count} windows, got {n}"
            )

        if split == "train":

            expected_ids = set(
                detail[
                    "train_ids"
                ]
            )

        else:

            expected_ids = set(
                self.split_table.loc[
                    self.split_table[
                        "split"
                    ] == split,
                    "recording_id",
                ].astype(
                    str
                )
            )

        actual_ids = set(
            recording_id.tolist()
        )

        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{profile}/{split}: recording membership changed"
            )

        metadata = (
            self.split_table
            .set_index(
                "recording_id"
            )
        )

        expected_labels = np.asarray(
            [
                int(
                    metadata.loc[
                        rid,
                        "class_id"
                    ]
                )
                for rid in recording_id
            ],
            dtype=np.int64,
        )

        if not np.array_equal(
            y_class,
            expected_labels,
        ):
            raise RuntimeError(
                f"{profile}/{split}: class labels disagree "
                "with frozen split metadata"
            )

        if (
            profile == "normal_only"
            and
            split == "train"
            and
            np.any(
                y_class != 0
            )
        ):
            raise RuntimeError(
                "normal_only/train contains fault windows"
            )


    def to_torch_dataset(
        self,
        prepared_split,
        target="class",
    ):

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is not installed. "
                "NumPy loading remains available."
            ) from exc

        if target not in {
            "class",
            "binary",
        }:
            raise ValueError(
                "target must be 'class' or 'binary'"
            )

        X = torch.from_numpy(
            prepared_split.X
        )

        if target == "class":
            y = torch.from_numpy(
                prepared_split.y_class
            )
        else:
            y = torch.from_numpy(
                prepared_split.y_binary
            )

        recording_id = (
            prepared_split
            .recording_id
        )

        start_sample = (
            prepared_split
            .start_sample
        )

        class WindowDataset(
            torch.utils.data.Dataset
        ):

            def __len__(self):
                return len(X)

            def __getitem__(self, index):

                return {
                    "X": X[index],
                    "y": y[index],

                    "recording_id":
                        str(
                            recording_id[
                                index
                            ]
                        ),

                    "start_sample":
                        int(
                            start_sample[
                                index
                            ]
                        ),
                }

        return WindowDataset()
