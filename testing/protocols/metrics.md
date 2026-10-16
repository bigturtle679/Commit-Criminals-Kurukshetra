# Metrics contract

For each patient and WT/TC/ET record TP, FP, FN, Dice=2TP/(2TP+FP+FN),
sensitivity=TP/(TP+FN), precision=TP/(TP+FP), predicted/reference volumes and volume error.
Use physical voxel volume from the affine determinant in mm3, divided by 1000 for mL;
validate units first. These are estimated volumes from segmentations.

Declare empty cases: Dice=1 when both masks are empty; Dice=0 when exactly one is empty.
Sensitivity is null when reference is empty; precision is null when prediction is empty.
Include denominators/counts in aggregate reporting; do not silently turn nulls into zero.
If HD95 is included, compute surface distance in millimeters using validated geometry.
Report empty-surface cases as undefined with a reason rather than inventing a distance.
Summarize per-patient values (mean/median and dispersion) and show all case-level results.
