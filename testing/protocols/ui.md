# Browser acceptance

After code readiness, start or refresh the server deliberately and record its revision.
Use verified local cases and the same settings as the CLI. Check model selection, patient
selection/uploads, queued/running/error states, and absence of concurrent GPU jobs.

Inspect overlays in axial/coronal/sagittal planes, native-space alignment and 3D meshes.
Verify report/download availability and uncertainty export if offered. Compare web and CLI
masks and metrics for the same case; explain differences rather than assuming parity.
Exercise incomplete modalities, invalid file type, corrupted volume, geometry mismatch and
missing trained weights. The UI must not display success or accuracy claims after failure.
Save screenshots with run/case identifiers locally; publish only appropriate compact evidence.
