"""Generate a synthetic 4-modality brain MRI phantom for verification."""
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/synthetic_demo'
DEST.mkdir(parents=True, exist_ok=True)

# Grid 96 x 96 x 96 at 1.0mm isotropic
shape = (96, 96, 96)
affine = np.diag([1.0, 1.0, 1.0, 1.0])

zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
center = (48, 48, 48)

# Brain mask (ellipsoid)
brain_dist = ((zz - center[0])/38)**2 + ((yy - center[1])/34)**2 + ((xx - center[2])/34)**2
brain = brain_dist <= 1.0

# Tumor center offset
t_center = (42, 52, 46)
t_dist = ((zz - t_center[0])/14)**2 + ((yy - t_center[1])/12)**2 + ((xx - t_center[2])/12)**2
wt_mask = t_dist <= 1.0
tc_mask = t_dist <= 0.6
et_mask = (t_dist <= 0.6) & (t_dist >= 0.3)
ed_mask = wt_mask & (~tc_mask)

# Discrete ground truth: 4: ET, 1: NET, 2: ED, 0: BG
gt = np.zeros(shape, dtype=np.uint8)
gt[ed_mask] = 2
gt[tc_mask] = 1
gt[et_mask] = 4

# Generate synthetic modalities
# FLAIR: high intensity in edema and tumor
flair = np.random.normal(10, 2, shape).astype(np.float32)
flair[brain] += 50
flair[ed_mask] += 120
flair[tc_mask] += 80
flair[~brain] = 0

# T1: baseline anatomical contrast
t1 = np.random.normal(10, 2, shape).astype(np.float32)
t1[brain] += 80
t1[wt_mask] -= 20
t1[~brain] = 0

# T1c: hyperintense enhancing rim
t1c = np.random.normal(10, 2, shape).astype(np.float32)
t1c[brain] += 80
t1c[et_mask] += 150
t1c[~brain] = 0

# T2: hyperintense edema and core
t2 = np.random.normal(10, 2, shape).astype(np.float32)
t2[brain] += 60
t2[ed_mask] += 140
t2[tc_mask] += 100
t2[~brain] = 0

# Save files
header = nib.Nifti1Header()
header.set_xyzt_units('mm')

for name, arr in [('t1n', t1), ('t1c', t1c), ('t2w', t2), ('t2f', flair)]:
    img = nib.Nifti1Image(arr, affine, header)
    nib.save(img, DEST / f"synthetic_{name}.nii.gz")

gt_img = nib.Nifti1Image(gt, affine, header)
nib.save(gt_img, DEST / "synthetic_seg.nii.gz")

print(f"Synthetic case generated at {DEST}")
print(f"WT volume: {wt_mask.sum()} voxels, ET volume: {et_mask.sum()} voxels")
