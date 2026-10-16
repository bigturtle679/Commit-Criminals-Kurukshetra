# NeuroScope 3D: research brief and 24-hour build plan

Prepared 16 October 2026. Sources checked on the web; no weights or datasets downloaded, and no inference benchmark run. This is a proposed implementation plan, not a completed application.

## Recommendation

Build a research prototype for preoperative adult glioma MRI: load four prepared modalities, segment with pretrained MONAI SegResNet on a cloud GPU, inspect synchronized slices and a 3D view, measure regions in mL, correct masks, and export reproducible results. Use React/Vite/TypeScript, NiiVue, FastAPI, PyTorch, MONAI and NiBabel. Do not train a model during the hackathon.

Assumptions: cloud CUDA GPU is available; public retrospective data only; four co-registered, skull-stripped modalities are required for inference. The audience is hackathon judges and research users. No real-time hospital integration, postoperative interpretation, diagnosis, survival prediction or treatment recommendations in v1.

## 1. Evidence and source decisions

| Resource | Finding | Decision |
|---|---|---|
| [MONAI production repository](https://huggingface.co/MONAI/brats_mri_segmentation) | Open-weight model bundle with configs and documentation | Primary model source; pin its revision and dependencies after smoke testing |
| [User-provided monai-test repository](https://huggingface.co/monai-test/brats_mri_segmentation) | Identified as testing integration; describes training data but is not itself a patient dataset or hosted inference API | Retain as a reference, use MONAI namespace for implementation |
| [Swin UNETR BraTS21](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21) | Official fold checkpoints and inference/tutorial resources available | Alternative model for later comparison, not an additional MVP dependency |
| [HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO) | Pretrained GPU inference; different output semantics | Secondary option, not interchangeable with the BraTS three-region interface |
| [Raidionics models](https://github.com/raidionics/Raidionics-models) | Released ONNX models for specific pre/postoperative tasks | Future task-specific extension; inspect each model's inputs and terms |

The selected MONAI bundle uses four channels ordered T1c, T1, T2, FLAIR and three sigmoid outputs ordered TC, WT, ET. Its documentation describes BraTS 2018 data, split into 200 training, 42 validation and 43 test studies. Published validation Dice is TC 0.8559, WT 0.9026, ET 0.7905; these are source-reported results, not our measurements. Its 16 GB training-memory note is not a measured minimum inference requirement. [Model documentation](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md)

### Papers for the presentation

| Paper | Why it belongs |
|---|---|
| [Myronenko, 2018](https://arxiv.org/abs/1810.11654) | Architecture motivation; describe MONAI as using a related approach, not an exact reproduction of the whole winning system |
| [Baid et al., BraTS 2021](https://arxiv.org/abs/2107.02314) | Multimodal benchmark, tumor-region definitions and evaluation context |
| [Simpson et al., 2019](https://arxiv.org/abs/1902.09063) and [Antonelli et al., MSD](https://arxiv.org/abs/2106.05735) | Dataset provenance and benchmark methodology |
| [Wang et al., TTA uncertainty](https://arxiv.org/abs/1807.07356) | Scientific basis for augmentation-derived uncertainty |
| [Swin UNETR, 2022](https://arxiv.org/abs/2201.01266) | Alternative segmentation architecture |
| [Jain et al., 2025](https://arxiv.org/html/2509.24973v1) | Recent research direction: training-time synthetic tumor augmentation and ensembling |

The Jain paper reports first place specifically in BraTS Lighthouse 2025 Task 1 Adult Glioma Segmentation. It combines three nnU-Net models and uses pretrained GliGAN generators during training. Generator weights are not the final segmentation ensemble. A downloadable checkpoint for that exact ensemble was not verified in this review. Its pre/post-treatment task includes resection cavities and differs from the older model's task. Do not directly compare its lesion-wise scores with older aggregate Dice results. Keep this as future research, not the hackathon implementation.

## 2. Data acquisition and testing strategy

| Source | Role | Access/caveat |
|---|---|---|
| [BraTS 2021 Kaggle mirror](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data) | Preferred demo and development cases | Page exists, but browser extraction did not expose its file inventory; verify actual files and terms after access |
| [BraTS 2021 official page](https://www.med.upenn.edu/cbica/brats2021/) | Provenance, task and access reference | Distinguish challenge-wide size from downloadable labelled subsets |
| [BraTS 2021 TCIA record](https://wiki.cancerimagingarchive.net/display/DOI/RSNA-ASNR-MICCAI-BraTS-2021) | Alternate access and patient-ID crosswalks | Processed challenge data and original DICOM have different access conditions; use the exact download's terms |
| [BraTS 2020 Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) and [official data description](https://www.med.upenn.edu/cbica/brats2020/data.html) | Labelled fallback/development data | Check downloaded inventory; a training-labelled dataset can support our testing only with overlap caveats |
| [BraTS 2018](https://www.med.upenn.edu/sbia/brats2018.html) | Reproduction and model compatibility reference | The selected model was trained on part of this release; do not treat all of it as unseen testing |
| [MSD Task01](https://medicaldecathlon.com/) and [AWS registry](https://registry.opendata.aws/msd/) | Primary access fallback | 484 training/266 test volumes; static dataset; CC BY-SA 4.0; public AWS access without an account |
| [BraTS Lighthouse 2025](https://www.synapse.org/Synapse%3Asyn64153130/) | Newer research track | Access/registration and different label definitions; not required for demo |
| [UCSF-PDGM paper](https://arxiv.org/abs/2109.00356) and [UPENN-GBM collection](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | Future generalization investigation | Public collections are not automatically independent of BraTS; investigate patient crosswalks before claiming external validation |

Acquisition sequence: try BraTS 2021 first; if access has not succeeded within 30 minutes, use MSD. Use BraTS 2020 if already available to the team. Start with three complete cases for integration. Expand to a fixed ten-case labelled evaluation subset after the pipeline works. Record case IDs, source, license, modality mapping and training-overlap status in a manifest. Verify checksums and preserve citations for redistributed demo assets.

Read MSD's dataset.json rather than assuming BraTS conventions: its multimodal 4D data and label mapping require an explicit adapter. Do not treat a fourth image dimension as time by default. For generic uploads, require four named 3D inputs.

Separate these claims:

- Development cases: used to debug; not an accuracy estimate.
- Fixed evaluation subset: labels used only for scoring after configuration is fixed; show every case's results, including failures. Independence from pretrained training data may remain unknown.
- Truly independent testing: requires patient-overlap verification. A later dataset year or a different repository is insufficient evidence.
- Official challenge test/validation images may have hidden labels; do not promise local Dice without available reference masks.
- “Live demo” means running inference on a newly selected public scan. None of these links supplies a real-time hospital feed.

## 3. Application and inference design

### User flow

Choose sample or upload T1/T1ce/T2/FLAIR → confirm modality assignments → validate inputs → queue inference → review axial/coronal/sagittal/3D views → inspect regional volumes → optionally correct mask → export.

Use one NiiVue multiplanar canvas with its render panel, modality switching, crosshairs, orientation labels, window/level, region visibility and overlay opacity. Use named API constants from the pinned package instead of hard-coded slice-type numbers. [Layouts](https://niivue.com/docs/layouts/) and [drawing tools](https://niivue.com/docs/drawing/) support this approach.

### Backend behavior

- FastAPI owns cases, an in-memory job queue and per-job files; one persistent GPU worker loads the model once and processes jobs serially. Return a job ID immediately; poll every two seconds. Restarted in-progress jobs become failed/retryable.
- Validate readable NIfTI data, finite intensities, non-singular affine, spatial units, four modalities, and compatible shape/affine. Geometry compatibility does not prove anatomical co-registration; require prepared inputs and visually inspect samples.
- Use the documented channel order and nonzero per-channel intensity normalization. Preserve the model's prepared coordinate convention. Do not blindly add RAS reorientation simply because another tutorial uses it. For v1, reject inputs outside the tested prepared geometry with a clear message; arbitrary raw-scan registration/skull stripping is deferred.
- Start with the bundle's inference settings: 240×240×160 sliding window, overlap 0.5, batch 1, GPU mixed precision. Measure memory and latency. If it runs out of memory, test a 128×128×128 window with the same overlap, record the change and recheck outputs before freezing configuration.
- Threshold sigmoid outputs at 0.5, use the bundle's ET→4, otherwise TC→1, otherwise WT→2 conversion, and derive displayed nested regions from the final label mask. Preserve original prediction separately from edits. [Inference config](https://huggingface.co/MONAI/brats_mri_segmentation/raw/main/configs/inference.json)
- Export the label mask aligned with the accepted input reference. If spatial preprocessing is introduced later, invert recorded transforms, use nearest-neighbor interpolation for discrete labels, and verify the output grid.
- Compute volume as N × abs(det(affine[:3,:3])) / 1000 after converting spatial units to millimetres. TC, WT and ET overlap; never add their volumes into a total. Report measurement of a predicted mask, not an exact biological tumor volume.

### Minimal interfaces

| Endpoint | Purpose |
|---|---|
| POST /api/cases | Four named modality files; return case ID and validation result |
| POST /api/cases/{id}/jobs | Queue single-pass inference; optional TTA only when implemented |
| GET /api/jobs/{id} | queued/running/succeeded/failed, stage, error and result URLs |
| GET /api/cases/{id}/results | Label map, region volumes, timings, provenance and optional evaluation metrics |
| POST /api/cases/{id}/edits | Corrected label NIfTI; validate grid/labels, save a separate revision and recalculate volumes |
| GET /api/cases/{id}/report | Download JSON and printable HTML report |

Serve NIfTI artifacts as files, not large JSON arrays. Reference masks are separate evaluation artifacts and are never model inputs. Keep public demo data only on the cloud instance. Apply upload/decompression limits and permit explicit case deletion. Add a concise research-use label.

## 4. Extra features in priority order

1. Input readiness panel: show modality assignment, geometry and preprocessing expectations; prevents silent invalid inference.
2. Auditable report: volume table, image snapshots, model revision, preprocessing version, threshold, inference time, dataset citation and original/edited status. Use printable HTML plus JSON; no LLM needed.
3. Reference comparison: toggle prediction/reference, display per-region Dice and absolute volume error for labelled demo cases.
4. Manual correction: paint/erase with undo; preserve original, export corrected NIfTI, recalculate measurements. NiiVue provides drawing primitives, but validate drawing import/export geometry explicitly.
5. Optional uncertainty review: identity plus single-axis flips along three spatial axes, unflip probabilities, average and calculate binary entropy per TC/WT/ET channel. Show selectable heatmaps and jump to high-entropy slices within the predicted region. Label it heuristic uncertainty, not calibrated error probability. Budget approximately four passes and benchmark actual latency. Freeze original single-pass results separately.

Future only: real paired longitudinal comparison, postoperative models, atlas-location summaries, missing-modality models, synthetic-data fine-tuning, multi-model ensembles. No simulated follow-up presented as real clinical progression.

### Existing projects from the notes

[Raidionics](https://github.com/raidionics/) is a verified precedent for segmentation and standardized reporting. [BodyMaps](https://github.com/astrohyo/BodyMaps) is a CT-viewer integration reference, not a brain-tumor model. The notes also mention a stroke demo, NeuroVis, Brainchop, niivue-streamlit, niivue-onnx and an ISCAS application: treat these as optional design leads, not validated dependencies. Their exact versions, licenses and suitability were not all verified. Direct NiiVue documentation is sufficient for the MVP.

## 5. Execution schedule and acceptance criteria

Suggested workstreams, if teammates are available: inference/data, viewer/UI, integration/evaluation. Otherwise follow the same milestones sequentially and cut optional features.

| Elapsed time | Work and completion gate |
|---|---|
| 0–2 h | Acquire three cases and weights; freeze source manifest; initialize GPU environment and viewer |
| 2–5 h | One successful inference with saved aligned mask, measured latency and volumes; viewer opens a sample with crosshairs |
| 5–9 h | Connect upload, job status and result loading; complete a genuine end-to-end case |
| 9–13 h | Add volume table, reference overlay and exportable report |
| 13–17 h | Add manual correction and test mask round-trip; if behind, keep report/reference comparison and defer editing |
| 17–20 h | Evaluate fixed subset; add TTA only if core checks pass and four-pass runtime is acceptable for the demo |
| 20–22 h | Validate errors, memory, browser behavior and cloud access; precompute labelled fallback cases |
| 22–24 h | Freeze features; rehearse, record a backup video and prepare citations/results slide |

Tests required before calling the MVP complete:

- Synthetic volume check: 1,000 voxels at 1×1×2 mm must produce 2 mL; reflected affine must remain positive; unknown units must not silently yield mL.
- Missing modality, corrupt file, incompatible affine, unsupported shape and GPU failure yield useful errors, not blank results.
- Asymmetric landmark/phantom verifies coordinate orientation, crosshairs and saved-mask alignment; editing/export/reload retains geometry.
- Assert exact channel mapping and dataset-specific label conversion; verify TC⊆WT and ET⊆TC in displayed metrics.
- Score all fixed evaluation cases by WT/TC/ET Dice and absolute volume error. For both masks empty, mark Dice N/A; if only one is empty, Dice is 0. Always report exclusions and sample counts.
- Show actual end-to-end latency, GPU model and peak memory. Do not advertise published benchmark speed as measured application speed.
- If cloud service fails, switch explicitly to “precomputed demo result”; do not present cached inference as a live run.

Deliverables: working browser demo, inference runner/API, pinned environments, dataset manifest, evaluation CSV, exported sample report, three cached cases where redistribution permits, and a cited presentation. The credible pitch is a geometry-aware, reviewable and reproducible workflow built around established segmentation research.
