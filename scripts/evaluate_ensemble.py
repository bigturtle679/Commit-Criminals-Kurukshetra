"""Patient-level evaluation infrastructure for NeuroScope 3D ensemble.

Evaluates multi-modal brain tumor segmentation on held-out patient cohorts
following the official BraTS evaluation protocol:
- Subregions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- Overlap metrics: Dice Similarity Coefficient (DSC), Sensitivity (Recall)
- Volumetric metrics: Absolute Volume Error (|Delta-V| mL)
- Patient-level separation: Enforces strict independent patient evaluation (no data leakage)

IMPORTANT:
- Synthetic phantoms are used ONLY for unit test and harness verification.
- Real segmentation performance requires authentic held-out cases and trained checkpoints.
- This evaluation harness is NOT for clinical diagnosis.
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from configs.config_loader import load_ensemble_config
from metrics import compute_dice, compute_sensitivity, calculate_volume_ml
from scripts.ensemble_inference import _sha256_file, run_ensemble
from scripts.models.model_loader import get_device, load_ensemble_models

logger = logging.getLogger('evaluate_ensemble')


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load and validate an evaluation manifest (JSON or CSV).

    Enforces patient-level separation by verifying unique patient IDs.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {path}")

    cases = []
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be an array of patient case objects.")
        cases = data
    elif path.suffix.lower() == '.csv':
        with open(path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cases = list(reader)
    else:
        raise ValueError(f"Unsupported manifest format '{path.suffix}'. Use .json or .csv.")

    required_keys = {'patient_id', 't1c', 't1', 't2', 'flair', 'expert_mask'}
    seen_patients = set()

    validated_cases = []
    for idx, c in enumerate(cases):
        missing = required_keys - set(c.keys())
        if missing:
            raise ValueError(f"Case {idx} in manifest missing required keys: {sorted(missing)}")

        pid = str(c['patient_id']).strip()
        if not pid:
            raise ValueError(f"Case {idx} has an empty patient_id.")
        if pid in seen_patients:
            raise ValueError(
                f"Data leakage risk: duplicate patient_id '{pid}' detected at index {idx}. "
                f"Evaluation requires strictly independent patient cases."
            )
        seen_patients.add(pid)
        validated_cases.append(c)

    return validated_cases


def evaluate_single_case(case: dict, ensemble_fn, out_dir: Path, expert_et: int = 4) -> dict:
    """Run inference and compute metrics on a single patient case.

    Validates spatial alignment before calculating metrics.
    """
    pid = case['patient_id']
    mri_paths = {
        't1c': Path(case['t1c']),
        't1': Path(case['t1']),
        't2': Path(case['t2']),
        'flair': Path(case['flair'])
    }
    expert_path = Path(case['expert_mask'])

    for k, p in mri_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} scan for patient {pid}: {p}")
    if not expert_path.exists():
        raise FileNotFoundError(f"Missing expert mask for patient {pid}: {expert_path}")

    # Spatial alignment verification
    t1c_img = nib.load(str(mri_paths['t1c']))
    gt_img = nib.load(str(expert_path))

    if gt_img.shape != t1c_img.shape:
        raise ValueError(
            f"Spatial mismatch for patient {pid}: expert mask shape {gt_img.shape} "
            f"does not match T1c shape {t1c_img.shape}."
        )
    if not np.allclose(gt_img.affine, t1c_img.affine, atol=1e-3):
        raise ValueError(
            f"Coordinate system mismatch for patient {pid}: expert mask affine "
            f"does not match T1c affine."
        )

    # Run inference
    case_out = out_dir / pid
    case_out.mkdir(parents=True, exist_ok=True)
    res = ensemble_fn(mri_paths=mri_paths, out_dir=case_out, expert_path=expert_path, expert_et=expert_et)

    pred_labels = res['_seg_data'] if '_seg_data' in res else nib.load(str(case_out / 'segmentation.nii.gz')).get_fdata()
    gt = gt_img.get_fdata()

    allowed = {0, 1, 2, expert_et}
    gt_labels = set(np.unique(gt).astype(int))
    if not gt_labels.issubset(allowed):
        raise ValueError(
            f"Patient {pid} expert mask contains unexpected labels: {gt_labels}. "
            f"Expected subset of {allowed}."
        )

    pred_regions = {
        'WT': np.isin(pred_labels, [1, 2, 4]),
        'TC': np.isin(pred_labels, [1, 4]),
        'ET': pred_labels == 4
    }
    gt_regions = {
        'WT': np.isin(gt, [1, 2, expert_et]),
        'TC': np.isin(gt, [1, expert_et]),
        'ET': gt == expert_et
    }

    metrics = {}
    for r in ['WT', 'TC', 'ET']:
        p_mask = pred_regions[r]
        g_mask = gt_regions[r]
        p_vol = calculate_volume_ml(p_mask, t1c_img.affine)
        g_vol = calculate_volume_ml(g_mask, t1c_img.affine)

        metrics[r] = {
            'dice': compute_dice(p_mask, g_mask),
            'sensitivity': compute_sensitivity(p_mask, g_mask),
            'pred_vol_ml': p_vol,
            'gt_vol_ml': g_vol,
            'abs_vol_err_ml': round(abs(p_vol - g_vol), 3)
        }

    return {
        'patient_id': pid,
        'metrics': metrics,
        'runtime_seconds': res.get('runtime_seconds', 0.0)
    }


def aggregate_metrics(case_results: list[dict]) -> dict:
    """Aggregate per-patient metrics into summary statistics."""
    summary = {}
    n_cases = len(case_results)
    if n_cases == 0:
        return {'sample_count': 0}

    for region in ['WT', 'TC', 'ET']:
        dices = [c['metrics'][region]['dice'] for c in case_results if c['metrics'][region]['dice'] is not None]
        sens = [c['metrics'][region]['sensitivity'] for c in case_results if c['metrics'][region]['sensitivity'] is not None]
        vol_errs = [c['metrics'][region]['abs_vol_err_ml'] for c in case_results]

        summary[region] = {
            'dice': {
                'mean': round(float(np.mean(dices)), 4) if dices else None,
                'median': round(float(np.median(dices)), 4) if dices else None,
                'std': round(float(np.std(dices)), 4) if dices else None,
                'min': round(float(np.min(dices)), 4) if dices else None,
                'max': round(float(np.max(dices)), 4) if dices else None,
            },
            'sensitivity': {
                'mean': round(float(np.mean(sens)), 4) if sens else None,
                'median': round(float(np.median(sens)), 4) if sens else None,
                'std': round(float(np.std(sens)), 4) if sens else None,
            },
            'abs_vol_err_ml': {
                'mean': round(float(np.mean(vol_errs)), 3) if vol_errs else None,
                'median': round(float(np.median(vol_errs)), 3) if vol_errs else None,
            }
        }

    summary['sample_count'] = n_cases
    return summary


def run_evaluation(manifest_path: str | Path, output_dir: str | Path,
                   config_path: str | Path | None = None, device: str = 'auto',
                   expert_et: int = 4, dry_run: bool = False,
                   custom_inference_fn=None) -> dict:
    """Execute complete patient cohort evaluation."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vcfg = load_ensemble_config(config_path)
    cases = load_manifest(manifest_path)

    if dry_run:
        logger.info("Dry-run mode: validated manifest syntax and %d patient cases.", len(cases))
        return {'status': 'DRY_RUN_PASSED', 'cases_validated': len(cases)}

    # Model and inference setup
    if custom_inference_fn is None:
        dev = get_device(device)
        models_bundle = load_ensemble_models(validated_config=vcfg, device=dev)
        def inference_fn(mri_paths, out_dir, expert_path, expert_et):
            return run_ensemble(
                mri_paths=mri_paths,
                out_dir=out_dir,
                validated_config=vcfg,
                expert_path=expert_path,
                expert_et=expert_et,
                device=device,
                models_bundle=models_bundle,
                update=lambda msg: None
            )
    else:
        inference_fn = custom_inference_fn

    results = []
    for c in cases:
        logger.info("Evaluating patient %s...", c['patient_id'])
        res = evaluate_single_case(c, inference_fn, out_dir, expert_et=expert_et)
        results.append(res)

    summary = aggregate_metrics(results)

    # Export CSV results
    csv_path = out_dir / 'evaluation_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'patient_id',
            'WT_dice', 'WT_sens', 'WT_pred_ml', 'WT_gt_ml',
            'TC_dice', 'TC_sens', 'TC_pred_ml', 'TC_gt_ml',
            'ET_dice', 'ET_sens', 'ET_pred_ml', 'ET_gt_ml'
        ])
        for r in results:
            m = r['metrics']
            writer.writerow([
                r['patient_id'],
                m['WT']['dice'], m['WT']['sensitivity'], m['WT']['pred_vol_ml'], m['WT']['gt_vol_ml'],
                m['TC']['dice'], m['TC']['sensitivity'], m['TC']['pred_vol_ml'], m['TC']['gt_vol_ml'],
                m['ET']['dice'], m['ET']['sensitivity'], m['ET']['pred_vol_ml'], m['ET']['gt_vol_ml'],
            ])

    # Export JSON summary report
    full_report = {
        'evaluation_type': 'patient_level_held_out',
        'status': 'COMPLETED',
        'ensemble_name': vcfg['ensemble_name'],
        'weights': {'model_a': vcfg['weight_a'], 'model_b': vcfg['weight_b']},
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'summary': summary,
        'per_patient_results': results,
        'effective_config': vcfg,
        'disclaimer': (
            'Research demonstration only. Not intended for primary clinical '
            'diagnosis or patient management. Metrics reflect performance '
            'on the evaluated cohort under the stated preprocessing settings.'
        )
    }

    json_path = out_dir / 'evaluation_summary.json'
    json_path.write_text(json.dumps(full_report, indent=2), encoding='utf-8')
    logger.info("Evaluation complete for %d cases. Results saved to %s", len(results), out_dir)
    return full_report


def main():
    p = argparse.ArgumentParser(description="NeuroScope 3D Patient-Level Ensemble Evaluation")
    p.add_argument("--manifest", required=True, help="Path to evaluation manifest (JSON or CSV)")
    p.add_argument("--output-dir", default="outputs/evaluation", help="Directory for output reports")
    p.add_argument("--config", default="configs/ensemble.json", help="Path to ensemble configuration")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--expert-et", type=int, default=4, help="ET label in expert reference (3 or 4)")
    p.add_argument("--dry-run", action="store_true", help="Validate manifest and inputs without model inference")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
        expert_et=args.expert_et,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
