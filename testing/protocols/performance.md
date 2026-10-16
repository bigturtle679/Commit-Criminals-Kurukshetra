# Runtime and repeatability

Record GPU name/VRAM, CPU/RAM, framework versions, precision, ROI size, sliding-window
settings and device. Measure cold end-to-end time including model loading separately from
warm inference. Synchronize CUDA around timed GPU operations; distinguish preprocessing,
inference, postprocessing and exports where instrumented.

Reset peak memory counters per measurement and record allocated/reserved peaks with units.
Record OOMs instead of quietly reducing ROI. A changed ROI/precision is a new configuration.
Repeat one case at least three times with fixed seeds/settings, comparing masks and metrics.
Report deterministic settings and any observed variation; do not infer accuracy from speed.
Do not run UI inference and terminal inference concurrently on the same laptop GPU.
