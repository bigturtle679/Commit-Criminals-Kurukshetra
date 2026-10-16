# Benchmark execution

Wait for an explicit readiness report. Freeze code revision, patient manifest and checkpoint
hashes; do not pull or edit runtime code while a run is active. Use one GPU job at a time.
Compare SwinUNETR, trained ResEnc alone and the fixed 0.8/0.2 probability ensemble on
exactly the same eligible patients and scoring masks. SegResNet may be an additional baseline.
If a checkpoint is unavailable, label that model blocked and run only ready baselines.

Save per-case results and failures. Show paired ensemble-minus-Swin changes for each region;
include regressions. Report precision and preprocessing differences. Do not optimize mixture
weights using final evaluation cases. A smaller valid baseline benchmark is preferable to
presenting an untrained component's scores as a model comparison.
Capture stdout/stderr and immutable run metadata under outputs/testing/<run_id>/.
