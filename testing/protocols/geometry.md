# MRI geometry

Accept original 3D NIfTI volumes (.nii or .nii.gz) for the current pipeline. Check readable
headers, finite voxel data, nonempty input, finite invertible affine, dimensions and spatial
units. Compare all four modality shapes, spacing and affines within documented tolerances.
Do not treat resampling as anatomical registration. Flag unregistered inputs.

Record any orientation/spacing transforms and restore predictions to the reference grid.
Use nearest-neighbor interpolation for categorical masks. Check output shape and affine
against the patient reference. Expert masks must match the evaluation grid; reject mismatch
or apply a documented, justified transform. Never silently compare different spaces.
Screenshots and slice arrays without reliable spatial metadata are ineligible for this path.
