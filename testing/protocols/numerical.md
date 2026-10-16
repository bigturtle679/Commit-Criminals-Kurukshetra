# Numerical checks

Run existing suites with the project .venv and capture exit status plus full logs:
`.venv/bin/python tests/test_ensemble.py`
`.venv/bin/python tests/test_numerical_stability.py`

Tests should exercise production functions, not just copies of formulas. Include entropy
at 0/1 and near bounds, extreme logits, empty region denominators, all-zero/nonfinite scans,
invalid geometry, missing checkpoints and repeated inference. Test FP32 plus the actual
production precision. FP16 entropy bounds need particular scrutiny because a value near
one may round to one even after a clamp intended for FP32.

Passing finite-parameter checks does not establish trained weights. Synthetic forward-pass
checks do not establish patient accuracy or stability for all scanner inputs. Report skipped
GPU checks explicitly, even if a script prints a fixed total passed count.
