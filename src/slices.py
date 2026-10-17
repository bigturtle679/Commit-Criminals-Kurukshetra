"""Render real MRI slices (with optional segmentation / uncertainty overlay) as PNG.

Used by the dashboard for the modality thumbnails, the slice filmstrip and split view.
Volumes are reoriented to canonical RAS so slice indices match NiiVue's RAS crosshair
(`dimsRAS`). Display convention matches the viewer: x (left→right of screen = patient
right→… as NiiVue draws it), anterior / superior towards the top.
"""
import io
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

MODALITIES = ('flair', 't1', 't1c', 't2')
AXES = {'sagittal': 0, 'coronal': 1, 'axial': 2}
# Backend labels: 1 non-enhancing core, 2 edema, 4 enhancing (same palette as the viewer).
LABEL_RGB = {1: (232, 117, 183), 2: (71, 199, 197), 4: (245, 166, 35)}


@lru_cache(maxsize=16)
def _canonical_uint8(path: str, mtime: float):
    """Load a volume in RAS order, windowed to uint8 (1st–99th percentile of brain voxels)."""
    data = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)
    data = np.nan_to_num(data)
    brain = data[data > 0]
    lo, hi = (np.percentile(brain, [1, 99]) if brain.size else (0.0, 1.0))
    scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return scaled.astype(np.uint8), data > 0


@lru_cache(maxsize=16)
def _canonical_raw(path: str, mtime: float):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj)


def _load(path: Path, raw=False):
    key = (str(path), path.stat().st_mtime)
    return _canonical_raw(*key) if raw else _canonical_uint8(*key)


def _plane(volume, axis: int, index: int):
    """2D plane oriented for display: rows top→bottom = superior/anterior → inferior/posterior."""
    plane = np.take(volume, index, axis=axis)
    # Remaining axes are (y, z), (x, z) or (x, y); put the second one vertical, flipped so
    # anterior/superior is at the top, and the first one horizontal.
    return np.flipud(plane.T)


def geometry(run_dir: Path):
    """Slice counts in RAS order, brain volume and the slices through the tumour centre."""
    ref = run_dir / 't1c.nii.gz'
    img = nib.as_closest_canonical(nib.load(str(ref)))
    dims = [int(d) for d in img.shape[:3]]
    voxel_ml = float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000.0
    _, brain_mask = _load(ref)
    out = {'dims_ras': dims, 'voxel_ml': voxel_ml, 'brain_ml': round(float(brain_mask.sum()) * voxel_ml, 2),
           'tumour_centre': None}
    seg_path = run_dir / 'segmentation.nii.gz'
    if seg_path.exists():
        seg = _load(seg_path, raw=True)
        idx = np.argwhere(seg > 0)
        if idx.size:
            out['tumour_centre'] = [int(round(v)) for v in idx.mean(axis=0)]
    return out


def render(run_dir: Path, modality: str, axis: str, index: int, overlay: str = 'none',
           size: int = 128, opacity: float = 0.55) -> bytes:
    if modality not in MODALITIES:
        raise ValueError('Unknown modality')
    if axis not in AXES:
        raise ValueError('Unknown axis')
    if overlay not in ('none', 'segmentation', 'uncertainty'):
        raise ValueError('Unknown overlay')
    size = int(min(max(size, 32), 512))
    path = run_dir / f'{modality}.nii.gz'
    if not path.exists():
        raise FileNotFoundError(f'{modality} is not available for this analysis')
    gray, _ = _load(path)
    ax = AXES[axis]
    index = int(min(max(index, 0), gray.shape[ax] - 1))
    rgb = np.repeat(_plane(gray, ax, index)[..., None], 3, axis=2).astype(np.float32)

    if overlay == 'segmentation' and (run_dir / 'segmentation.nii.gz').exists():
        seg = _load(run_dir / 'segmentation.nii.gz', raw=True)
        if seg.shape[:3] == gray.shape[:3]:
            labels = _plane(seg, ax, index)
            for label, colour in LABEL_RGB.items():
                mask = labels == label
                rgb[mask] = rgb[mask] * (1 - opacity) + np.array(colour, np.float32) * opacity
    elif overlay == 'uncertainty' and (run_dir / 'uncertainty.nii.gz').exists():
        unc = _load(run_dir / 'uncertainty.nii.gz', raw=True).astype(np.float32)
        if unc.shape[:3] == gray.shape[:3]:
            u = np.clip(_plane(unc, ax, index), 0, 1)[..., None]
            warm = np.concatenate([np.ones_like(u) * 255, u * 200 + 40, np.zeros_like(u) + 30], axis=2)
            alpha = np.clip(u * 1.6, 0, 0.85)
            rgb = rgb * (1 - alpha) + warm * alpha

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w = image.size[1], image.size[0]
    scale = size / max(h, w)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()
