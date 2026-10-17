"""Export the logged ResEnc-M training curves for the dashboard (no smoothing, no invented points).

  .venv/bin/python scripts/export_training_curves.py --run work/resenc_curves --out src/web/training_curves.json
"""
import argparse, json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    loss = read_jsonl(args.run / 'loss_history.jsonl')
    val = read_jsonl(args.run / 'val_history.jsonl')
    split = json.loads((args.run / 'split.json').read_text())
    data = {
        'description': (f"Logged during a ResEnc-M training run: {len(split['train'])} BraTS 2021 training patients, "
                        f"{len(split['val'])} validation patients, {loss[-1]['iteration'] if loss else 0} steps (batch 2, 96³ patches), "
                        "same data split and recipe as the deployed checkpoint."),
        'loss': [{'iteration': r['iteration'], 'loss': round(r['loss'], 5)} for r in loss],
        'val': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in val],
    }
    args.out.write_text(json.dumps(data, indent=1))
    print(f'wrote {args.out}: {len(data["loss"])} loss points, {len(data["val"])} validation points')


if __name__ == '__main__':
    main()
