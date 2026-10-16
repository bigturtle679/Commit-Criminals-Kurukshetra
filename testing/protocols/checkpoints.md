# Checkpoint readiness

For each model record provenance, architecture/configuration, checkpoint path and SHA-256,
training dataset/split if available, input order and output meanings. Check state loading
for missing/unexpected keys and incompatible shapes. Explain any intentionally excluded
training-only keys; never silently accept missing inference weights.

Missing trained weights must stop inference before reporting predictions. Randomly initialized
networks can serve architecture tests only. Custom ResEnc architecture must be matched to
its actual training checkpoint; do not substitute official nnU-Net weights by name alone.

Current blocker: no ResEnc checkpoint was found in configured/fallback paths at inspection.
Swin checkpoint state keys matched the instantiated architecture. This verifies loading,
not segmentation quality. Recheck all paths/hashes before each frozen run.
