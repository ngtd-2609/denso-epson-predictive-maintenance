"""Verify files, restore frozen Epson arrays, and check the data loader.

Prepared arrays are checked against their original content signatures.
This entry point does not change the split, scaler or scientific protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from data_pipeline import prepare
from loader_v1 import EpsonPreparedLoader

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_signature(value):
    # Canonical little-endian bytes, including Unicode IDs, for portability.
    value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<")))
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def verify_arrays(folder, expected):
    for relative, details in expected.items():
        path = folder / relative
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(details["arrays"]):
                raise ValueError(f"Unexpected arrays: {path}")
            for key, signature in details["arrays"].items():
                if array_signature(archive[key]) != signature:
                    raise ValueError(f"Array content differs: {path}:{key}")


def rebase_rows(rows, raw):
    """Use frozen recording IDs, never the original machine's absolute paths."""
    result = []
    for row in rows:
        rid = row["recording_id"]
        if not rid or any(c in rid for c in "/\\:") or rid in (".", ".."):
            raise ValueError("Invalid recording ID")
        result.append({**row, "filepath": str((raw / f"{rid}.csv").resolve())})
    return result


def verify_release():
    repo = ROOT.parent
    manifest = read_json(repo / "release_manifest.json")
    for relative, expected in manifest["files"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()):
            raise ValueError("Manifest path escapes repository")
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Release file changed: {relative}")
    print(f"PASS: {len(manifest['files'])} release files")


def restore(raw):
    verify_release()
    prepared = ROOT / "results" / "prepared_v1"
    expected = read_json(ROOT / "prepared_arrays_manifest.json")["files"]
    if any((prepared / name).exists() for name in expected):
        raise FileExistsError("Prepared arrays already exist; use verify --data. No overwrite.")
    rows = rebase_rows(read_json(ROOT / "results/split_seed42/split.json"), raw)
    # prepare validates every raw-file checksum before processing.
    with tempfile.TemporaryDirectory(prefix="epson_restore_", dir=ROOT / "results") as work:
        generated = Path(work) / "prepared"
        report = prepare(rows, generated, [1, 2, 4], 3000, 101)
        if report != read_json(prepared / "preparation_report.json"):
            raise ValueError("Reproduction differs from frozen preparation report")
        for profile in ("normal_only", "k1", "k2", "k4"):
            if read_json(generated / profile / "scaler.json") != read_json(prepared / profile / "scaler.json"):
                raise ValueError(f"Scaler changed: {profile}")
            if file_hash(generated / profile / "windows.csv") != file_hash(prepared / profile / "windows.csv"):
                raise ValueError(f"Window metadata changed: {profile}")
        verify_arrays(generated, expected)
        # Only publish after all profiles passed. Interrupted copies are caught by verify --data.
        for relative in expected:
            destination = prepared / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (generated / relative).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
    print("PASS: 12 prepared files restored with exact frozen array contents")


def smoke():
    loader = EpsonPreparedLoader(ROOT)
    expected_counts = {"normal_only": (752, 8), "k1": (1222, 13),
                       "k2": (1692, 18), "k4": (2632, 28)}
    for profile, (count, records) in expected_counts.items():
        data = loader.load(profile, "train")
        if data.X.shape != (count, 3, 3000) or len(set(data.recording_id)) != records:
            raise ValueError(f"Unexpected training data: {profile}")
        if len(loader.get_scaler(profile)["fit_recording_ids"]) != records:
            raise ValueError(f"Unexpected scaler: {profile}")
        print(f"PASS {profile}/train: {data.X.shape}, {records} recordings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--data", action="store_true")
    rebuild = commands.add_parser("restore")
    rebuild.add_argument("--raw", type=Path, required=True)
    commands.add_parser("smoke")
    args = parser.parse_args()
    if args.command == "verify":
        verify_release()
        if args.data:
            verify_arrays(ROOT / "results/prepared_v1",
                          read_json(ROOT / "prepared_arrays_manifest.json")["files"])
            print("PASS: all 12 prepared files have exact frozen array contents")
    elif args.command == "restore":
        restore(args.raw)
    else:
        smoke()


if __name__ == "__main__":
    main()
