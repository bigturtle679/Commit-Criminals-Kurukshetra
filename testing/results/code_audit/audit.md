# Code audit — 2026-10-16

## Implemented
- Missing explicitly selected checkpoints fail instead of silently using random weights or a different checkpoint. Both ensemble paths are checked before GPU allocation.
- Strict checkpoint state loading rejects incompatible keys/shapes.
- CLI and web use one shared ensemble predictor, including per-model ROI, overlap, channel order, temperature and threshold.
- The previously ignored minimum-ET-volume setting is now zero by default; nonzero values fail as unsupported rather than pretending to apply filtering.
- Shared expert-mask validation rejects mismatched geometry, nonfinite/fractional/unknown labels and permits valid empty masks.
- Shared production metrics add precision, TP/FP/FN and volume errors. Empty-reference sensitivity and empty-prediction precision are null; empty/empty Dice is one.
- FP16/BF16 entropy is promoted to FP32 and nonfinite inputs rejected.
- Web uncertainty files use float32 and are downloadable. CPU report generation no longer assumes CUDA.
- Hash-verified patient-manifest runner saves per-case results, failures, checkpoint hashes, source revision and runtime metadata.

## Evidence and limitations
See tracked regression/stability logs in testing/results/code_audit/. The 2D local training experiment does not supply weights for the 3D custom ResEnc architecture.
Standalone ResEnc and full ensemble remain blocked. Their stated input order cannot be certified against missing training artifacts. Synthetic CLI/web parity does not establish trained ensemble accuracy.
The original source provenance of existing local demo cases was not re-established; overlap is unknown. Unknown NIfTI units are still treated as mm by the existing pipeline; use documented mm inputs for benchmark claims.
HD95 is not implemented. No official challenge evaluator or clinical validation is claimed. Actual browser interaction remains Antigravity's task; the parity test calls the production web pipeline directly.
The export_uncertainty_map config flag is not honored as a disable switch: uncertainty is always exported. Report this effective behavior.
Do not accept a ready baseline as evidence that the full ensemble is ready.
