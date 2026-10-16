# Pilot cohort

Start with three complete cases for pipeline debugging. Freeze a separate pilot selection
of twenty unique eligible labeled patients using seed 42 before inspecting predictions.
Sort eligible patient IDs before random selection and save selected IDs and source versions.
Twenty cases are a practical pilot size, not a claim of statistical or clinical adequacy.

A case is eligible only with all four volumes, a documented expert mask, valid geometry,
and successful integrity checks. Record exclusions and reasons, including incomplete files.
Report fewer cases honestly if twenty cannot be acquired. Do not replace poor-scoring
cases. Keep tuning/debugging cases identified separately and never tune on final scores.
