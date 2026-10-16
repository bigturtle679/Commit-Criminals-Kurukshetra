# Codex assignment

Work on branch model-testing in the shared DI-02 repository. Read AGENTS.md and this
folder's protocols. Own model/app correctness, manifest validation, evaluation infrastructure
and CPU regression tests. Antigravity owns data acquisition and GPU/browser benchmarking.
Do not touch its scripts/data or run GPU jobs while it is benchmarking.

First fix missing-weight behavior and verify checkpoint compatibility. Then audit input/output
semantics, CLI/web parity, geometry/label validation and production metric tests. Implement
a manifest-driven evaluation entry point consistent with the tracked schemas. Publish actual
commands and per-model readiness to outputs/testing/benchmark_ready.md and copy a compact
reviewable summary to testing/ANALYSIS.md. Never fabricate data, scores or trained checkpoints.

Commit meaningful tested changes and append analysis with evidence. Only one agent performs
Git branch/index/commit/push operations at a time; Antigravity hands off its changes for a
coordinated commit unless explicitly assigned a separate worktree. No branch switching during
shared work. Twenty setup commits are not twenty completed model tests.
