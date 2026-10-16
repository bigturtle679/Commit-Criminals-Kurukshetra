# Dataset acquisition

Antigravity owns downloading; Codex must reuse the same data. Inspect disk space,
existing files, source inventory and archive sizes before transfer. Use official Kaggle
API tooling or kagglehub in a separate environment under `work/data_acquisition/`.
See https://github.com/Kaggle/kagglehub for download and authentication documentation.

Attempt public access first. Consent/login may be required; perform it locally and
keep tokens outside the repository. Never write credentials to reports or terminal logs.
Set download/cache locations under `data/external/`. Pin and record the resolved version.
Prefer selecting complete individual cases if exposed in the inventory. A monolithic
archive may require downloading the whole archive even to extract a few patients.
Account for archive plus extracted size; report impractical downloads before starting.

Reuse validated files. Verify successful transfer and integrity; do not assume interrupted
archives are complete. Begin with three complete cases; expand to a pilot of twenty.
No automatic download has been executed as part of these instructions.
