# Preprocessing audit

Compare the actual training/inference specification with each model's executed path:
modality order, canonical orientation, voxel spacing, intensity normalization, foreground
crop, ROI size, overlap, padding, blending and precision. Persist effective values rather
than copying config fields that may be ignored.

Current code orders Swin inputs as FLAIR,T1c,T1,T2 and custom ResEnc as FLAIR,T1,T1c,T2.
The latter requires confirmation from its training configuration. Compare CLI and web paths
with identical input cases. Account for interpolation and cropping when restoring outputs.
Review unused configuration entries such as temperature and minimum ET volume before claiming
those options are applied. Retain the reference mask only for the final scoring stage.
