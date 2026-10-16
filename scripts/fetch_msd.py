"""Fetch a small reproducible subset of the fixed public uncompressed MSD TAR.

HTTP Range avoids downloading unselected patients. Archive size and tar checksums
are validated. Cached prefix/tail files from interrupted downloads may be reused.
Only regular files with selected, known names are written; no extractall().
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tarfile

URL = "https://msd-for-monai.s3.us-west-2.amazonaws.com/Task01_BrainTumour.tar"
SIZE = 7608266240
TAIL_START = SIZE - 32 * 1024**2
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def fetch_range(start, size, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    headers = target.with_suffix(target.suffix + ".headers")
    subprocess.run([
        "curl", "-sSL", "--fail", "--retry", "3", "--max-time", "240",
        "--range", f"{start}-{start + size - 1}", "-D", str(headers),
        "-o", str(partial), URL,
    ], check=True)
    expected = f"content-range: bytes {start}-{start + size - 1}/{SIZE}"
    if expected not in headers.read_text().lower() or partial.stat().st_size != size:
        raise ValueError("Server did not return the exact pinned archive byte range")
    partial.replace(target)


def header(block):
    return tarfile.TarInfo.frombuf(block, "utf-8", "strict")


def split_cases(cases, seed):
    cases = list(cases)
    random.Random(seed).shuffle(cases)
    n = len(cases)
    if n < 10:
        raise ValueError("Use at least 10 cases for a patient-level split")
    train = int(n * .7)
    val = int(n * .2)
    for i, case in enumerate(cases):
        case["split"] = "train" if i < train else "val" if i < train + val else "test"
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 10 <= args.cases <= 484:
        p.error("--cases must be between 10 and 484")
    cache = ROOT / "work/downloads"
    dest = ROOT / "work/data/msd"
    dest.mkdir(parents=True, exist_ok=True)
    prefix = cache / "Task01_BrainTumour.tar"

    def read_range(start, size):
        if prefix.exists() and prefix.stat().st_size >= start + size:
            with prefix.open("rb") as f:
                f.seek(start)
                return f.read(size)
        target = cache / "ranges" / f"{start}-{size}.bin"
        fetch_range(start, size, target)
        return target.read_bytes()

    # Metadata and selected images occur at the start of this pinned archive.
    offset, selected, metadata = 0, [], None
    while len(selected) < args.cases:
        entry = header(read_range(offset, 512))
        start = offset + 512
        if entry.name == "Task01_BrainTumour/dataset.json":
            raw = read_range(start, entry.size)
            metadata = json.loads(raw)
            (dest / "dataset.json").write_bytes(raw)
            assert metadata["modality"] == {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}
            assert metadata["labels"] == {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        if entry.isfile() and entry.name.startswith("Task01_BrainTumour/imagesTr/") and entry.name.endswith(".nii.gz"):
            name = Path(entry.name).name
            image = dest / "imagesTr" / name
            image.parent.mkdir(exist_ok=True)
            if not image.exists() or image.stat().st_size != entry.size:
                image.write_bytes(read_range(start, entry.size))
            selected.append(name)
            print(f"Image {len(selected)}/{args.cases}: {name}", flush=True)
        offset = start + ((entry.size + 511) // 512) * 512
        if offset >= SIZE or (metadata is None and offset > 1024**2):
            raise ValueError("Unexpected archive layout; stop rather than guess")

    tail_path = cache / "msd-tail.bin"
    fetch_range(TAIL_START, SIZE - TAIL_START, tail_path)
    tail = tail_path.read_bytes()
    masks = {}
    for i in range(0, len(tail) - 511, 512):
        try:
            entry = header(tail[i:i + 512])
        except (tarfile.HeaderError, UnicodeError, ValueError):
            continue
        name = Path(entry.name).name
        if entry.isfile() and entry.name.startswith("Task01_BrainTumour/labelsTr/") and name in selected:
            data = tail[i + 512:i + 512 + entry.size]
            if len(data) != entry.size:
                raise ValueError("Truncated mask")
            path = dest / "labelsTr" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            masks[name] = path
    if set(masks) != set(selected):
        raise ValueError("Not all selected masks found in archive tail")
    cases = []
    for name in selected:
        image = dest / "imagesTr" / name
        mask = masks[name]
        cases.append({"id": name.removesuffix(".nii.gz"), "image": str(image.relative_to(ROOT)),
                      "label": str(mask.relative_to(ROOT)), "image_sha256": sha(image),
                      "label_sha256": sha(mask)})
    manifest = {"dataset": "msd", "source": URL, "license": metadata["licence"],
                "selection": "first N training images in pinned official TAR, shuffled for splits",
                "seed": args.seed, "channel_order": ["flair", "t1", "t1ce", "t2"],
                "label_schema": "msd", "pretraining_overlap": "unknown",
                "cases": split_cases(cases, args.seed)}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {len(cases)} labelled cases at {dest}", flush=True)


if __name__ == "__main__":
    main()
