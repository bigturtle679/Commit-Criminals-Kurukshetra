"""Four-channel 2D baseline, data preparation, short benchmark and explicit training.

Run from the repository root. Full training only occurs with the `train` command.
Resized-slice metrics are development metrics, not native-grid clinical validation.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

ROOT = Path(__file__).resolve().parents[1]


def region_masks(labels, schema):
    allowed = {0, 1, 2, 3} if schema == "msd" else {0, 1, 2, 4}
    if not set(np.unique(labels)).issubset(allowed):
        raise ValueError(f"Unexpected {schema} labels")
    core, enhancing = (2, 3) if schema == "msd" else (1, 4)
    return np.stack([(labels == core) | (labels == enhancing), labels > 0, labels == enhancing])


def build_model():
    return UNet(spatial_dims=2, in_channels=4, out_channels=3,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1)


def load_case(case, schema):
    paths = case.get("images")
    if paths:
        scans = [nib.load(str(ROOT / x)) for x in paths]
        ref = scans[0]
        if any(s.shape != ref.shape or not np.allclose(s.affine, ref.affine, atol=1e-4, rtol=0) for s in scans):
            raise ValueError(f"Misaligned modalities: {case['id']}")
        image = np.stack([s.get_fdata(dtype=np.float32) for s in scans])
    else:
        ref = nib.load(str(ROOT / case["image"]))
        if len(ref.shape) != 4 or ref.shape[-1] != 4:
            raise ValueError("MSD must contain four channels in its last axis")
        # MSD FLAIR,T1,T1ce,T2 -> local model T1ce,T1,T2,FLAIR.
        image = np.moveaxis(ref.get_fdata(dtype=np.float32), -1, 0)[[2, 1, 3, 0]]
    label_img = nib.load(str(ROOT / case["label"]))
    if image.shape[1:] != label_img.shape or not np.allclose(ref.affine, label_img.affine, atol=1e-4, rtol=0):
        raise ValueError(f"Misaligned mask: {case['id']}")
    label = label_img.get_fdata(dtype=np.float32)
    if not np.isfinite(image).all() or not np.isfinite(label).all():
        raise ValueError("Nonfinite data")
    for c in range(4):
        nz = image[c] != 0
        if not nz.any() or image[c][nz].std() < 1e-8:
            raise ValueError("Empty or constant input modality")
        image[c][nz] = (image[c][nz] - image[c][nz].mean()) / image[c][nz].std()
    return image, region_masks(label, schema)


def prepare(manifest_path, size):
    manifest = json.loads(manifest_path.read_text())
    out = ROOT / "work/prepared" / manifest["dataset"] / str(size)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for case in manifest["cases"]:
        x_path, y_path = out / f"{case['id']}_x.npy", out / f"{case['id']}_y.npy"
        # Always regenerate so a stale cache cannot silently outlive changed inputs.
        x, y = load_case(case, manifest["label_schema"])
        x = torch.from_numpy(x.transpose(3, 0, 1, 2).copy())
        y = torch.from_numpy(y.transpose(3, 0, 1, 2).astype(np.float32))
        xr = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        yr = F.interpolate(y, size=(size, size), mode="nearest")
        np.save(x_path, xr.numpy().astype(np.float16))
        np.save(y_path, yr.numpy().astype(np.uint8))
        items.append({"id": case["id"], "split": case["split"], "slices": len(xr),
                      "image": str(x_path.relative_to(ROOT)), "label": str(y_path.relative_to(ROOT))})
        print(f"Prepared {case['id']}", flush=True)
    result = {"source_manifest": str(manifest_path), "size": size, "cases": items,
              "metric_space": "resized axial slices, not native geometry"}
    (out / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out / "index.json")


class Slices(Dataset):
    def __init__(self, cases):
        self.arrays = [(np.load(ROOT / c["image"], mmap_mode="r"),
                        np.load(ROOT / c["label"], mmap_mode="r")) for c in cases]
        self.indices = [(i, k) for i, c in enumerate(cases) for k in range(c["slices"])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, k = self.indices[index]
        x, y = self.arrays[i]
        return torch.from_numpy(np.array(x[k], dtype=np.float32)), torch.from_numpy(np.array(y[k], dtype=np.float32))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def choose_device(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; benchmark in a local Terminal or select cpu explicitly")
    return torch.device(name)


def validate(model, cases, device, batch_size):
    scores = []
    model.eval()
    with torch.inference_mode():
        for case in cases:
            tp, pred_n, true_n = [torch.zeros(3, dtype=torch.float64) for _ in range(3)]
            for x, y in DataLoader(Slices([case]), batch_size=batch_size, num_workers=0):
                pred = (torch.sigmoid(model(x.to(device))) > .5).cpu()
                truth = y.bool()
                tp += (pred & truth).sum((0, 2, 3))
                pred_n += pred.sum((0, 2, 3))
                true_n += truth.sum((0, 2, 3))
            denom = pred_n + true_n
            dice = [float(2 * tp[i] / denom[i]) if denom[i] else None for i in range(3)]
            scores.append({"case_id": case["id"], "dice_TC_WT_ET": dice})
    vals = [v for s in scores for v in s["dice_TC_WT_ET"] if v is not None]
    return scores, sum(vals) / len(vals) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "benchmark", "train"])
    p.add_argument("--config", default="configs/local_training.json")
    p.add_argument("--index", default="work/prepared/msd/128/index.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.command == "prepare":
        prepare(ROOT / cfg["manifest"], cfg["patch_size"])
        return
    if args.steps < 1:
        p.error("--steps must be positive")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.set_num_threads(4)
    index = json.loads((ROOT / args.index).read_text())
    groups = {s: [c for c in index["cases"] if c["split"] == s] for s in ["train", "val", "test"]}
    if any(not cs for cs in groups.values()):
        raise ValueError("Train/val/test must all contain patients")
    device = choose_device(args.device)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True)
    loader = DataLoader(Slices(groups["train"]), batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    batches = iter(loader)

    def step():
        nonlocal batches
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if args.command == "benchmark":
        model.train()
        for _ in range(3):
            step()
        sync(device)
        start = time.perf_counter()
        losses = [step() for _ in range(args.steps)]
        sync(device)
        seconds = (time.perf_counter() - start) / args.steps
        report = {"device": str(device), "torch": torch.__version__, "steps": args.steps,
                  "seconds_per_training_step": seconds, "batch_size": cfg["batch_size"],
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "training_only_hours_20_epochs_200_steps": seconds * 4000 / 3600,
                  "excludes": ["data preparation", "validation", "thermal slowdown", "checkpoint writing"],
                  "last_loss": losses[-1], "patient_counts": {k: len(v) for k, v in groups.items()}}
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    run = out / f"unet2d_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir()
    best = -math.inf
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = [step() for _ in range(cfg["steps_per_epoch"])]
        scores, score = validate(model, groups["val"], device, cfg["batch_size"])
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "mean_validation_dice": score, "cases": scores}
        with (run / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        checkpoint = {"state_dict": model.cpu().state_dict(), "config": cfg, "index": index, "epoch": epoch + 1}
        torch.save(checkpoint, run / "last.pt")
        if score is not None and score > best:
            best = score
            torch.save(checkpoint, run / "best.pt")
        model.to(device)
        print(f"Epoch {epoch + 1}: validation Dice={score}", flush=True)
    # Test patients remain untouched during model selection.
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True)["state_dict"])
    scores, score = validate(model, groups["test"], device, cfg["batch_size"])
    (run / "test.json").write_text(json.dumps({"mean_dice": score, "cases": scores}, indent=2) + "\n")


if __name__ == "__main__":
    main()
