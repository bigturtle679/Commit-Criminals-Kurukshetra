"""Slice thumbnails and geometry used by the dashboard (real data only, safe inputs)."""
import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from src import slices


@pytest.fixture()
def run_dir(tmp_path):
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])            # LPS-like, as in BraTS files
    brain = np.zeros((20, 24, 16), np.float32)
    brain[3:17, 4:20, 2:14] = 100.0
    seg = np.zeros(brain.shape, np.uint8)
    seg[5:9, 6:10, 4:8] = 2
    seg[6:8, 7:9, 5:7] = 4
    for m in ('t1', 't1c', 't2', 'flair'):
        nib.save(nib.Nifti1Image(brain, affine), tmp_path / f'{m}.nii.gz')
    nib.save(nib.Nifti1Image(seg, affine), tmp_path / 'segmentation.nii.gz')
    nib.save(nib.Nifti1Image(np.where(seg > 0, 0.6, 0.0).astype(np.float32), affine), tmp_path / 'uncertainty.nii.gz')
    return tmp_path


def test_geometry_reports_ras_dims_brain_volume_and_tumour_centre(run_dir):
    g = slices.geometry(run_dir)
    assert g['dims_ras'] == [20, 24, 16]
    assert g['brain_ml'] == pytest.approx(14 * 16 * 12 / 1000, abs=0.01)
    # Canonical RAS flips x and y of the LPS input: index i -> 19 - i, j -> 23 - j.
    assert g['tumour_centre'] == [round(19 - 6.5), round(23 - 7.5), round(5.5)]


@pytest.mark.parametrize('axis, shape', [('axial', (24, 20)), ('coronal', (16, 20)), ('sagittal', (16, 24))])
def test_render_plane_shapes(run_dir, axis, shape):
    png = slices.render(run_dir, 'flair', axis, 5, size=512)
    img = Image.open(io.BytesIO(png))
    h, w = shape
    scale = 512 / max(h, w)
    assert img.size == (round(w * scale), round(h * scale))


def test_segmentation_overlay_colours_tumour_pixels_only(run_dir):
    plain = np.asarray(Image.open(io.BytesIO(slices.render(run_dir, 'flair', 'axial', 5, 'none', 64))))
    over = np.asarray(Image.open(io.BytesIO(slices.render(run_dir, 'flair', 'axial', 5, 'segmentation', 64))))
    changed = np.any(plain != over, axis=2)
    assert changed.any() and changed.mean() < 0.2


@pytest.mark.parametrize('kwargs', [{'modality': '../etc/passwd'}, {'axis': 'oblique'}, {'overlay': 'expert'}])
def test_render_rejects_unknown_inputs(run_dir, kwargs):
    args = {'modality': 'flair', 'axis': 'axial', 'overlay': 'none'} | kwargs
    with pytest.raises(ValueError):
        slices.render(run_dir, args['modality'], args['axis'], 3, args['overlay'])


def test_missing_modality_is_reported(run_dir):
    (run_dir / 't2.nii.gz').unlink()
    with pytest.raises(FileNotFoundError):
        slices.render(run_dir, 't2', 'axial', 3)


def test_index_is_clamped_to_volume(run_dir):
    assert slices.render(run_dir, 'flair', 'axial', 9999, size=32)
