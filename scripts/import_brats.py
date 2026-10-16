"""Index authorized BraTS 2021/2020 NIfTI cases already inside work/data/.

This never accepts site terms or downloads gated data on the user's behalf.
"""
import argparse
import json
from pathlib import Path
from fetch_msd import ROOT, sha, split_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--dataset", choices=["brats2021", "brats2020"], default="brats2021")
    p.add_argument("--cases", type=int, default=50)
    args = p.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT / "work/data"):
        p.error("Keep datasets inside the project work/data directory")
    cases = []
    for seg in sorted(root.rglob("*_seg.nii.gz")):
        stem = seg.name.removesuffix("_seg.nii.gz")
        images = [seg.with_name(f"{stem}_{m}.nii.gz") for m in ["t1ce", "t1", "t2", "flair"]]
        if not all(x.is_file() for x in images):
            continue
        cases.append({"id": stem, "images": [str(x.relative_to(ROOT)) for x in images],
                      "label": str(seg.relative_to(ROOT)), "label_sha256": sha(seg),
                      "image_sha256": [sha(x) for x in images]})
        if len(cases) == args.cases:
            break
    if len(cases) < args.cases:
        p.error(f"Only {len(cases)} complete labelled cases found; requested {args.cases}")
    manifest = {"dataset": args.dataset, "label_schema": "brats", "seed": 42,
                "channel_order": ["t1ce", "t1", "t2", "flair"], "pretraining_overlap": "unknown",
                "cases": split_cases(cases, 42)}
    dest = root / "manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
