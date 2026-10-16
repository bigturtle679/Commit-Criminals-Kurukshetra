"""Reference-mask validation shared by CLI and web inference."""
import nibabel as nib
import numpy as np


def read_expert(path, reference, enhancing):
    if enhancing not in (3, 4):
        raise ValueError('Expert ET label must be 3 or 4')
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=0):
        raise ValueError('Expert mask must share the original MRI grid and affine')
    if image.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
        raise ValueError('Expert spatial units must be millimeters')
    data = image.get_fdata()
    if not np.isfinite(data).all() or not np.isin(data, [0, 1, 2, enhancing]).all():
        raise ValueError('Expert mask labels do not match the selected convention')
    return image
