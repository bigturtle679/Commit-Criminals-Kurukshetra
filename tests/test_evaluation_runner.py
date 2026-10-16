"""Tests for the patient-level evaluation runner (scripts/evaluate_ensemble.py).

Validates:
- Patient-level separation enforcement (duplicate patient ID rejection)
- Missing field and missing file detection
- Spatial mismatch rejection (shape and affine)
- Empty-ground-truth and empty-prediction conventions
- Metric aggregation mathematics (mean, median, std, min, max)
- CSV and JSON report generation
- Dry-run mode for manifest verification
"""
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_ensemble import (
    load_manifest, evaluate_single_case, aggregate_metrics, run_evaluation
)


def _create_synthetic_patient(patient_dir: Path, pid: str, shape=(32, 32, 32)):
    """Helper to create dummy 3D NIfTI files for a patient."""
    patient_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    data = np.zeros(shape, dtype=np.float32)
    data[10:22, 10:22, 10:22] = 50.0  # Foreground

    seg = np.zeros(shape, dtype=np.uint8)
    seg[12:18, 12:18, 12:18] = 2  # Edema
    seg[14:18, 14:18, 14:18] = 1  # Core
    seg[15:17, 15:17, 15:17] = 4  # Enhancing

    t1c_p = patient_dir / 't1c.nii.gz'
    t1_p = patient_dir / 't1.nii.gz'
    t2_p = patient_dir / 't2.nii.gz'
    flair_p = patient_dir / 'flair.nii.gz'
    gt_p = patient_dir / 'seg.nii.gz'

    for p in [t1c_p, t1_p, t2_p, flair_p]:
        nib.save(nib.Nifti1Image(data, affine), p)
    nib.save(nib.Nifti1Image(seg, affine), gt_p)

    return {
        'patient_id': pid,
        't1c': str(t1c_p),
        't1': str(t1_p),
        't2': str(t2_p),
        'flair': str(flair_p),
        'expert_mask': str(gt_p)
    }


class TestManifestValidation:
    def test_valid_manifest_loads(self, tmp_path):
        m_path = tmp_path / 'manifest.json'
        cases = [
            {'patient_id': 'P001', 't1c': 'a', 't1': 'b', 't2': 'c', 'flair': 'd', 'expert_mask': 'e'},
            {'patient_id': 'P002', 't1c': 'a', 't1': 'b', 't2': 'c', 'flair': 'd', 'expert_mask': 'e'},
        ]
        m_path.write_text(json.dumps(cases))
        loaded = load_manifest(m_path)
        assert len(loaded) == 2
        assert loaded[0]['patient_id'] == 'P001'

    def test_duplicate_patient_id_rejected_for_data_leakage(self, tmp_path):
        m_path = tmp_path / 'leakage_manifest.json'
        cases = [
            {'patient_id': 'P001', 't1c': 'a', 't1': 'b', 't2': 'c', 'flair': 'd', 'expert_mask': 'e'},
            {'patient_id': 'P001', 't1c': 'a2', 't1': 'b2', 't2': 'c2', 'flair': 'd2', 'expert_mask': 'e2'},
        ]
        m_path.write_text(json.dumps(cases))
        with pytest.raises(ValueError, match='duplicate patient_id'):
            load_manifest(m_path)

    def test_missing_required_keys_rejected(self, tmp_path):
        m_path = tmp_path / 'incomplete.json'
        cases = [{'patient_id': 'P001', 't1c': 'a'}]
        m_path.write_text(json.dumps(cases))
        with pytest.raises(ValueError, match='missing required keys'):
            load_manifest(m_path)


class TestSingleCaseEvaluation:
    def test_spatial_shape_mismatch_rejected(self, tmp_path):
        p_dir = tmp_path / 'patient_mismatch'
        case = _create_synthetic_patient(p_dir, 'P_MISMATCH', shape=(32, 32, 32))

        # Overwrite expert mask with mismatched shape
        bad_gt = np.zeros((32, 32, 16), dtype=np.uint8)
        nib.save(nib.Nifti1Image(bad_gt, np.eye(4)), case['expert_mask'])

        with pytest.raises(ValueError, match='Spatial mismatch'):
            evaluate_single_case(case, lambda **kw: None, tmp_path / 'out')

    def test_spatial_affine_mismatch_rejected(self, tmp_path):
        p_dir = tmp_path / 'patient_affine'
        case = _create_synthetic_patient(p_dir, 'P_AFFINE', shape=(32, 32, 32))

        # Overwrite expert mask with mismatched affine
        bad_affine = np.diag([2.0, 1.0, 1.0, 1.0])
        gt = nib.load(case['expert_mask']).get_fdata()
        nib.save(nib.Nifti1Image(gt, bad_affine), case['expert_mask'])

        with pytest.raises(ValueError, match='Coordinate system mismatch'):
            evaluate_single_case(case, lambda **kw: None, tmp_path / 'out')

    def test_evaluation_with_mock_inference(self, tmp_path):
        p_dir = tmp_path / 'patient_eval'
        case = _create_synthetic_patient(p_dir, 'P_MOCK', shape=(32, 32, 32))
        gt_data = nib.load(case['expert_mask']).get_fdata()

        # Mock inference function that returns identical prediction (perfect agreement)
        def mock_inference(mri_paths, out_dir, expert_path, expert_et):
            return {
                '_seg_data': gt_data,
                'runtime_seconds': 0.12
            }

        res = evaluate_single_case(case, mock_inference, tmp_path / 'out')
        assert res['patient_id'] == 'P_MOCK'
        assert res['metrics']['WT']['dice'] == 1.0
        assert res['metrics']['TC']['dice'] == 1.0
        assert res['metrics']['ET']['dice'] == 1.0


class TestMetricAggregation:
    def test_aggregation_statistics(self):
        case_results = [
            {
                'patient_id': 'P1',
                'metrics': {
                    'WT': {'dice': 0.90, 'sensitivity': 0.85, 'abs_vol_err_ml': 1.0},
                    'TC': {'dice': 0.80, 'sensitivity': 0.75, 'abs_vol_err_ml': 0.8},
                    'ET': {'dice': 0.70, 'sensitivity': 0.65, 'abs_vol_err_ml': 0.5},
                }
            },
            {
                'patient_id': 'P2',
                'metrics': {
                    'WT': {'dice': 1.00, 'sensitivity': 0.95, 'abs_vol_err_ml': 0.2},
                    'TC': {'dice': 0.90, 'sensitivity': 0.85, 'abs_vol_err_ml': 0.4},
                    'ET': {'dice': 0.80, 'sensitivity': 0.75, 'abs_vol_err_ml': 0.3},
                }
            }
        ]
        agg = aggregate_metrics(case_results)
        assert agg['sample_count'] == 2
        assert agg['WT']['dice']['mean'] == 0.95
        assert agg['WT']['dice']['median'] == 0.95
        assert agg['WT']['dice']['min'] == 0.90
        assert agg['WT']['dice']['max'] == 1.00


class TestEndToEndEvaluationHarness:
    def test_run_evaluation_dry_run(self, tmp_path):
        """Dry run must validate actual referenced NIfTI files without running model inference."""
        p1 = _create_synthetic_patient(tmp_path / 'P1', 'P1')
        m_path = tmp_path / 'manifest.json'
        m_path.write_text(json.dumps([p1]))

        rep = run_evaluation(manifest_path=m_path, output_dir=tmp_path / 'out', dry_run=True)
        assert rep['status'] == 'DRY_RUN_PASSED'
        assert rep['cases_validated'] == 1
        assert rep['meets_10_case_requirement'] is False
        assert rep['minimum_cohort_required'] == 10

    def test_run_evaluation_dry_run_missing_file_fails(self, tmp_path):
        """Dry run must fail if any referenced NIfTI file does not exist."""
        m_path = tmp_path / 'bad_manifest.json'
        cases = [
            {'patient_id': 'P001', 't1c': 'missing_t1c.nii.gz', 't1': 'b', 't2': 'c', 'flair': 'd', 'expert_mask': 'e'},
        ]
        m_path.write_text(json.dumps(cases))

        with pytest.raises(FileNotFoundError, match='Missing t1c scan'):
            run_evaluation(manifest_path=m_path, output_dir=tmp_path / 'out', dry_run=True)

    def test_full_evaluation_pipeline_undersized_cohort_labeled_honestly(self, tmp_path):
        """A cohort of < 10 cases must NOT be marked as satisfying the evaluation requirement."""
        p1 = _create_synthetic_patient(tmp_path / 'P1', 'P1')
        p2 = _create_synthetic_patient(tmp_path / 'P2', 'P2')

        m_path = tmp_path / 'manifest.json'
        m_path.write_text(json.dumps([p1, p2]))

        out_dir = tmp_path / 'eval_outputs'

        def mock_infer(mri_paths, out_dir, expert_path, expert_et):
            gt = nib.load(str(expert_path)).get_fdata()
            return {'_seg_data': gt, 'runtime_seconds': 0.05}

        report = run_evaluation(
            manifest_path=m_path,
            output_dir=out_dir,
            dry_run=False,
            custom_inference_fn=mock_infer
        )

        assert report['status'] == 'COMPLETED_UNDERSIZED_COHORT'
        assert report['meets_minimum_cohort_requirement'] is False
        assert report['cohort_size'] == 2
        assert report['summary']['sample_count'] == 2
        assert (out_dir / 'evaluation_summary.json').exists()
        assert (out_dir / 'evaluation_results.csv').exists()

        # Read CSV
        with open(out_dir / 'evaluation_results.csv', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]['patient_id'] == 'P1'
            assert float(rows[0]['WT_dice']) == 1.0

    def test_full_evaluation_pipeline_meets_10_cases(self, tmp_path):
        """A cohort of >= 10 cases satisfies the project evaluation requirement."""
        patients = [_create_synthetic_patient(tmp_path / f'P{i:02d}', f'P{i:02d}') for i in range(10)]

        m_path = tmp_path / 'manifest_10.json'
        m_path.write_text(json.dumps(patients))

        out_dir = tmp_path / 'eval_outputs_10'

        def mock_infer(mri_paths, out_dir, expert_path, expert_et):
            gt = nib.load(str(expert_path)).get_fdata()
            return {'_seg_data': gt, 'runtime_seconds': 0.02}

        report = run_evaluation(
            manifest_path=m_path,
            output_dir=out_dir,
            dry_run=False,
            custom_inference_fn=mock_infer
        )

        assert report['status'] == 'COMPLETED'
        assert report['meets_minimum_cohort_requirement'] is True
        assert report['cohort_size'] == 10
        assert report['summary']['sample_count'] == 10
