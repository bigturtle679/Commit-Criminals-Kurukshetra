"""Rebuild the 3D display meshes of saved analyses with the current mesh settings.

Uses only each run's saved segmentation and FLAIR; volumes and metrics are untouched.
  .venv/bin/python scripts/rebuild_meshes.py [work/runs]
"""
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline import mesh  # noqa: E402

runs = Path(sys.argv[1] if len(sys.argv) > 1 else 'work/runs')
done = 0
for run in sorted(p for p in runs.iterdir() if (p / 'segmentation.nii.gz').exists()):
    seg_img = nib.load(str(run / 'segmentation.nii.gz'))
    seg = np.asarray(seg_img.dataobj)
    if (run / 'flair.nii.gz').exists():
        flair = nib.load(str(run / 'flair.nii.gz'))
        mesh(np.asarray(flair.dataobj) != 0, flair.affine, run / 'brain.obj', smooth_sigma=2.0, step_size=2)
    for name, labels in (('tumor', [1, 2, 4]), ('core', [1, 4]), ('et', [4])):
        target = run / f'{name}.obj'
        if not mesh(np.isin(seg, labels), seg_img.affine, target):
            target.unlink(missing_ok=True)
    done += 1
print(f'rebuilt meshes for {done} analyses')
