"""Input validation and reference-mask verification for NeuroScope 3D.

Enforces:
- File uniqueness (rejects duplicate filenames across MRI modality uploads).
- NIfTI 3D geometry, affine coordinate consistency, millimeter units.
- Non-constant intensities and 1 GB file size limits.
- Expert mask shape/affine matching and label convention checking.
- Quality warnings for suspicious foreground occupancy.
"""
from pathlib import Path
import nibabel as nib
import numpy as np

MAX_FILE_BYTES = 1_000_000_000  # 1 GB


class InputValidationError(ValueError):
    """Raised when uploaded MRI files fail structural, filename, or geometry validation."""
    pass


def validate_upload_filenames(files: dict) -> None:
    """Ensure that all 4 MRI modality uploads have distinct filenames.

    A valid multi-sequence examination requires 4 distinct MRI acquisitions (T1, T1c, T2, FLAIR).
    Uploading the same file or files with identical filenames across different modalities
    is strictly rejected.

    Args:
        files: Mapping of modality roles ('t1c', 't1', 't2', 'flair', optional 'expert')
               to UploadFile objects, Path objects, or filename strings.

    Raises:
        InputValidationError: If duplicate filenames are detected among uploads.
    """
    seen = {}
    for role, f in files.items():
        if f is None:
            continue
        # Extract filename from FastAPI UploadFile, Path, or str
        raw_name = getattr(f, 'filename', None)
        if raw_name is None:
            raw_name = getattr(f, 'name', None) if not isinstance(f, (str, Path)) else str(f)
        if not raw_name:
            continue
        base_name = Path(str(raw_name)).name.strip()
        norm_name = base_name.lower()
        if not norm_name:
            continue
        if norm_name in seen:
            prev_role, prev_name = seen[norm_name]
            raise InputValidationError(
                f"Duplicate filename '{base_name}' detected for {prev_role.upper()} and {role.upper()}. "
                "All 4 MRI modality uploads must be distinct files with unique filenames."
            )
        seen[norm_name] = (role, base_name)


def read_image(path, is_expert_mask=False):
    """Validate and load a NIfTI MRI volume or expert reference mask.

    Args:
        path: Path or path-like object pointing to a NIfTI file.
        is_expert_mask: If True, skips MRI-specific intensity contrast checks.

    Returns:
        nibabel.Nifti1Image: The validated NIfTI image.
    """
    path = Path(path)

    # 1 GB file size limit check
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'File exceeds 1 GB limit: {path.name} ({path.stat().st_size / 1e9:.1f} GB)')

    try:
        img = nib.load(str(path))
    except Exception as e:
        raise ValueError(f'Cannot read {path.name} as NIfTI: {e}') from e

    if len(img.shape) != 3:
        raise ValueError(
            f'{path.name} is {len(img.shape)}D (shape {img.shape}). '
            f'Each upload must be a single 3D MRI volume, not a 2D image or 4D series.'
        )
    if min(img.shape) < 8:
        raise ValueError(f'{path.name} has shape {img.shape} which is too small for brain MRI.')
    if np.prod(img.shape) > 35_000_000:
        raise ValueError(f'{path.name} exceeds 35 million voxels ({np.prod(img.shape):,}).')

    units = img.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError(f'{path.name}: spatial units must be millimeters, got {units!r}.')

    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError(f'{path.name}: invalid spatial affine in MRI header.')

    a = img.get_fdata(dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f'{path.name} contains NaN or Inf values.')

    if not is_expert_mask:
        if not np.any(a):
            raise ValueError(f'{path.name} is entirely zero/blank.')
        if np.std(a[a != 0]) < 1e-6:
            raise ValueError(f'{path.name} has constant intensity (no tissue contrast).')

    return img


def read_expert(path, reference, enhancing=4):
    """Validate and load an expert reference ground-truth mask."""
    if enhancing not in (3, 4):
        raise ValueError('Expert ET label must be 3 or 4')
    image = nib.load(str(path))
    if image.shape != reference.shape:
        raise InputValidationError(
            f'Expert mask shape {image.shape} does not match reference shape {reference.shape}'
        )
    if not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=0):
        raise InputValidationError(
            'Expert mask must share the original MRI grid and spatial affine'
        )
    if image.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
        raise ValueError('Expert spatial units must be millimeters')
    data = image.get_fdata()
    if not np.isfinite(data).all() or not np.isin(data, [0, 1, 2, enhancing]).all():
        raise ValueError('Expert mask labels do not match the selected convention')
    return image


def validate_study(paths: dict, expert_path=None, enhancing: int = 4):
    """Validate all four MRI sequence files and optional expert reference mask.

    Checks:
    - Distinct filenames across all sequence paths.
    - 3D spatial geometry, affine consistency, finite voxel intensities.
    - Quality warnings (such as suspicious foreground occupancy > 60%).
    - Reference mask alignment and valid label conventions.

    Returns:
        tuple: (dict_of_loaded_images, list_of_quality_warnings)
    """
    validate_upload_filenames(paths)

    warnings = []
    imgs = {}
    for k in ('t1c', 't1', 't2', 'flair'):
        if k not in paths:
            raise InputValidationError(f"Missing required MRI sequence: {k.upper()}")
        imgs[k] = read_image(paths[k])

    ref = imgs['t1c']
    for k, img in imgs.items():
        if k == 't1c':
            continue
        if img.shape != ref.shape:
            raise InputValidationError(f"Shape mismatch: {k.upper()} shape {img.shape} does not match T1C {ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-3, rtol=0):
            raise InputValidationError(f"Affine mismatch: {k.upper()} spatial coordinate system does not match T1C")

    # Foreground occupancy warning check
    for k, img in imgs.items():
        data = img.get_fdata(dtype=np.float32)
        total_voxels = data.size
        nonzero_count = np.count_nonzero(data > 0)
        occupancy = nonzero_count / float(total_voxels) if total_voxels > 0 else 0
        if occupancy > 0.60:
            warnings.append({
                'code': 'suspicious_foreground',
                'modality': k,
                'message': f"{k.upper()} has high foreground occupancy ({occupancy*100:.1f}% > 60%), suggesting background noise."
            })

    if expert_path is not None:
        expert_img = read_expert(expert_path, ref, enhancing=enhancing)
        imgs['expert'] = expert_img

    return imgs, warnings
