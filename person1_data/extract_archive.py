"""Extract only validated Epson CSV members; zipfile checks each member CRC."""
import argparse
import json
import zipfile
from pathlib import Path

from data_pipeline import NAME, digest, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
with zipfile.ZipFile(args.archive) as archive:
    files = [i for i in archive.infolist() if not i.is_dir()]
    if len(files) != 96 or any(not NAME.fullmatch(Path(i.filename).name) for i in files):
        raise ValueError('Unexpected archive contents; inspect before extraction')
    if len({Path(i.filename).name for i in files}) != len(files):
        raise ValueError('Duplicate filenames')
    args.out.mkdir(parents=True)
    entries = []
    for info in files:
        # Flatten known filenames: never extract untrusted paths or symlinks.
        raw = archive.read(info)
        target = args.out / Path(info.filename).name
        target.write_bytes(raw)
        entries.append(dict(filename=target.name, bytes=len(raw), crc32=info.CRC, crc_checked=True))
    write_json(args.out / 'archive_integrity.json', dict(
        source_url='https://www.epsondevice.com/sensing/en/dataset/zip/rotorkit_dataset_trimmed.zip',
        archive_sha256=digest(args.archive), archive_bytes=args.archive.stat().st_size, entries=entries))
print(json.dumps(dict(extracted=len(entries), crc_checked=len(entries))))
