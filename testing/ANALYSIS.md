# Model testing analysis

Date: 2026-10-16 (Asia/Kolkata)
Baseline reviewed: `9e8b90c2e7031ff4e41ab4d784e85855c03b0c46`
Status: SETUP COMPLETE; ACCURACY BENCHMARK PENDING.

## Observed in this session
- GitHub baseline introduces an 80% SwinUNETR / 20% custom ResEnc ensemble.
- Both existing standalone suites previously completed: ensemble 6/6;
  numerical stability 6/6. These are synthetic/software checks, not accuracy evidence.
  Their raw console output was not saved to a file; rerun with captured logs for publication.
- The project environment detected an NVIDIA RTX 4060 Laptop GPU.
- SwinUNETR checkpoint loading was audited: no missing or unexpected state keys.
- ResEnc checkpoints were absent from the configured and fallback locations when checked.
- The loader currently permits randomly initialized models if checkpoints are missing.
  Full ensemble accuracy testing is blocked until this is fixed and trained weights supplied.
- The three existing GLI reference masks contain labels 0,1,2,3. Their ET setting is 3.
  This mapping must not be copied blindly to other BraTS releases.

## What has not been established
No new dataset has been downloaded by this branch setup. No new patient inference,
ensemble accuracy score, training-overlap clearance, or clinical validation is claimed.
Previously quoted 98.2% / 95.9% scores are not verified ensemble benchmark results.
Custom ResEnc code is not evidence of compatibility with official nnU-Net checkpoints.

## Pending report
Record dataset versions, case manifest hash, code commit, checkpoint hashes, runtime
settings, per-patient WT/TC/ET metrics, failures and paired baseline comparisons.
Report uncertainty about training overlap. Keep tuning cases separate from evaluation.
Do not replace this status with PASS based solely on unit tests.

## Update policy
Append dated run summaries with evidence paths. Retain failed runs and original metrics.
Link compact tracked results under `testing/results/`; raw artifacts remain local under
`outputs/testing/`. State explicitly when an artifact is only available locally.
