"""Train the custom 3D ResEnc-M UNet on BraTS 2021 Task 1 for the 8:2 ensemble.

Preprocessing, channel order and output regions mirror scripts/ensemble_inference.py
exactly, so the saved checkpoint loads with scripts/models/model_loader.load_resenc
(strict=True) and plugs into configs/ensemble.json as model_b.

  input  : canonical RAS 1 mm volumes, channels [flair, t1, t1c, t2],
           MONAI NormalizeIntensity(nonzero=True, channel_wise=True), foreground crop
  output : 3 sigmoid channels [TC, WT, ET]; TC={1,4}, WT={1,2,4}, ET={4}

The 20 pilot benchmark patients are excluded from training and validation so they
remain held out for ResEnc. Usage:

  .venv/bin/python scripts/train_resenc.py --data data/external/brats2021_task1 \
      --out work/resenc_training --iters 60000
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.models.resenc_unet import ResEncUNet  # noqa: E402

CHANNELS = ['flair', 't1', 't1ce', 't2']  # ResEnc order from configs/ensemble.json
PATCH = (96, 96, 96)
PILOT_MANIFEST = ROOT / 'outputs/testing/antigravity/manifest.json'
NORM = NormalizeIntensity(nonzero=True, channel_wise=True)


def load_case(case_dir):
    """Load one case exactly as inference does; return (image[4,...], regions[3,...])."""
    pid = case_dir.name
    imgs = [nib.as_closest_canonical(nib.load(case_dir / f'{pid}_{m}.nii.gz')) for m in CHANNELS]
    seg_img = nib.as_closest_canonical(nib.load(case_dir / f'{pid}_seg.nii.gz'))
    for img in imgs + [seg_img]:
        if img.shape != imgs[0].shape or not np.allclose(img.affine, imgs[0].affine, atol=1e-3):
            raise ValueError(f'{pid}: geometry mismatch')
        if not np.allclose(nib.affines.voxel_sizes(img.affine), 1, atol=1e-3):
            raise ValueError(f'{pid}: not 1 mm isotropic')
    raw = np.stack([i.get_fdata(dtype=np.float32) for i in imgs])
    seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    if not np.isin(seg, [0, 1, 2, 4]).all():
        raise ValueError(f'{pid}: unexpected labels {np.unique(seg)}')
    pos = np.where(np.any(raw != 0, axis=0))
    sl = tuple(slice(max(0, int(p.min()) - 4), min(raw.shape[i + 1], int(p.max()) + 5)) for i, p in enumerate(pos))
    image = np.asarray(NORM(raw))[(slice(None),) + sl]
    seg = seg[sl]
    regions = np.stack([np.isin(seg, [1, 4]), np.isin(seg, [1, 2, 4]), seg == 4]).astype(np.uint8)
    return np.ascontiguousarray(image, dtype=np.float32), regions


class PatchDataset(Dataset):
    """Random 96^3 patches; one third are forced to contain tumour (nnU-Net-style oversampling)."""

    def __init__(self, cases, length):
        self.cases, self.length = cases, length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        image, regions = load_case(self.cases[rng.integers(len(self.cases))])
        pad = [(0, 0)] + [(0, max(0, p - s)) for p, s in zip(PATCH, image.shape[1:])]
        image, regions = np.pad(image, pad), np.pad(regions, pad)
        shape = image.shape[1:]
        tumour = np.argwhere(regions[1]) if rng.random() < 1 / 3 else np.empty((0, 3))
        if len(tumour):
            c = tumour[rng.integers(len(tumour))]
            start = [int(np.clip(c[i] - PATCH[i] // 2, 0, shape[i] - PATCH[i])) for i in range(3)]
        else:
            start = [int(rng.integers(0, shape[i] - PATCH[i] + 1)) for i in range(3)]
        sl = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(start, PATCH))
        x, y = image[sl].copy(), regions[sl].copy()
        for axis in (1, 2, 3):
            if rng.random() < 0.5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        brain = x != 0
        x = x * rng.uniform(0.9, 1.1, (4, 1, 1, 1)).astype(np.float32) + rng.uniform(-0.1, 0.1, (4, 1, 1, 1)).astype(np.float32)
        x = np.where(brain, x, 0).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def soft_dice_loss(logits, target, eps=1e-5):
    p = torch.sigmoid(logits.float())
    dims = (0, 2, 3, 4)
    inter = (p * target).sum(dims)
    return 1 - ((2 * inter + eps) / (p.sum(dims) + target.sum(dims) + eps)).mean()


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


@torch.no_grad()
def validate(model, cases, device):
    model.eval()
    scores = []
    for case in cases:
        image, regions = load_case(case)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = sliding_window_inference(torch.from_numpy(image)[None].to(device), PATCH, 2, model,
                                              overlap=0.5, mode='gaussian')
        pred = (torch.sigmoid(logits[0].float()) >= 0.5).cpu().numpy()
        scores.append([dice(pred[i], regions[i].astype(bool)) for i in range(3)])
    model.train()
    return np.mean(scores, axis=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data/external/brats2021_task1')
    ap.add_argument('--out', type=Path, default=ROOT / 'work/resenc_training')
    ap.add_argument('--iters', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--val-cases', type=int, default=40)
    ap.add_argument('--test-cases', type=int, default=50, help='random patients held out for final testing')
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=100, help='average and record training loss every N iterations')
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / 'train.log').open('a')

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + '\n'); log.flush()

    all_cases = sorted(d for d in args.data.iterdir() if d.is_dir() and d.name.startswith('BraTS2021_'))
    all_cases = [d for d in all_cases if all((d / f'{d.name}_{m}.nii.gz').is_file() for m in CHANNELS + ['seg'])]
    held_out = {p['patient_id'] for p in json.loads(PILOT_MANIFEST.read_text())['patients']}
    pool = [d for d in all_cases if d.name not in held_out]
    random.Random(args.seed).shuffle(pool)
    test = pool[:args.test_cases]
    val, train = pool[args.test_cases:args.test_cases + args.val_cases], pool[args.test_cases + args.val_cases:]
    split_file = args.out / 'split.json'
    if split_file.exists():
        split = json.loads(split_file.read_text())
        by_name = {d.name: d for d in all_cases}
        train, val = [by_name[n] for n in split['train']], [by_name[n] for n in split['val']]
        test = [by_name[n] for n in split.get('test', [])]
    else:
        split_file.write_text(json.dumps({'seed': args.seed, 'held_out_pilot': sorted(held_out),
                                          'test': [d.name for d in test], 'train': [d.name for d in train], 'val': [d.name for d in val]}, indent=1))
    say(f'cases: {len(all_cases)} complete, train {len(train)}, val {len(val)}, test {len(test)}, held-out pilot {len(held_out)}')
    assert not ({d.name for d in train + val} & (held_out | {d.name for d in test})), 'held-out case leaked into training'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = ResEncUNet(in_channels=4, out_channels=3, preset='M').to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    start_it, best = 0, -1.0
    latest = args.out / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start_it, best = ck['iteration'], ck['best_mean_dice']
        say(f'resumed at iteration {start_it}, best mean Dice {best:.4f}')

    loader = DataLoader(PatchDataset(train, (args.iters - start_it) * args.batch), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    model.train()
    t0, run_loss = time.time(), 0.0
    for it, (x, y) in enumerate(loader, start=start_it + 1):
        lr = args.lr * (1 - (it - 1) / args.iters) ** 0.9
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y) + soft_dice_loss(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at iteration {it}')
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(opt); scaler.update()
        run_loss += loss.item()

        if it % args.log_every == 0:
            el = time.time() - t0
            rate = el / (it - start_it)
            mean_loss = run_loss / args.log_every
            with (args.out / 'loss_history.jsonl').open('a') as f:
                f.write(json.dumps({'iteration': it, 'loss': mean_loss, 'lr': lr}) + '\n')
            if it % 100 == 0 or args.log_every >= 100:
                say(f'it {it}/{args.iters} loss {mean_loss:.4f} lr {lr:.5f} {rate:.3f}s/it '
                    f'ETA {(args.iters - it) * rate / 3600:.2f}h')
            run_loss = 0.0
        if it % args.save_every == 0 or it == args.iters:
            torch.save({'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
                        'iteration': it, 'best_mean_dice': best}, latest)
        if it % args.val_every == 0 or it == args.iters:
            wt_tc_et = validate(model, val, device)
            mean = float(np.mean(wt_tc_et))
            say(f'VAL it {it}: TC {wt_tc_et[0]:.4f} WT {wt_tc_et[1]:.4f} ET {wt_tc_et[2]:.4f} mean {mean:.4f}')
            with (args.out / 'val_history.jsonl').open('a') as f:
                f.write(json.dumps({'iteration': it, 'TC': wt_tc_et[0], 'WT': wt_tc_et[1], 'ET': wt_tc_et[2], 'mean': mean}) + '\n')
            if mean > best:
                best = mean
                torch.save({'state_dict': model.state_dict(), 'iteration': it, 'val_dice': {'TC': wt_tc_et[0], 'WT': wt_tc_et[1], 'ET': wt_tc_et[2]},
                            'preset': 'M', 'channel_order': ['flair', 't1', 't1c', 't2'], 'output_order': ['TC', 'WT', 'ET'],
                            'patch': PATCH, 'split': str(split_file)}, args.out / 'best.pt')
                say(f'new best checkpoint (mean {best:.4f})')
            torch.save({'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
                        'iteration': it, 'best_mean_dice': best}, latest)
        if it >= args.iters:
            break
    say('training complete')


if __name__ == '__main__':
    main()
