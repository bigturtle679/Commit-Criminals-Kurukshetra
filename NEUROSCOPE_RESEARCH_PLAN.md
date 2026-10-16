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

### Planned MRI inputs, file formats and tumor outputs

We will support four complementary MRI contrasts: T1, T1Gd/T1ce, T2 and FLAIR. The upload form, dataset adapters and model inputs will use these names consistently.

| Correct name | Common aliases | What it contributes | Project use |
|---|---|---|---|
| T1-weighted, before contrast | T1, T1w, t1n, precontrast T1 | Anatomical structure and a baseline for comparison with postcontrast imaging | Structural viewing and one input channel |
| T1-weighted, after gadolinium contrast | T1Gd, T1ce, T1c, T1-CE, postcontrast T1 | Highlights enhancing regions associated with contrast leakage/blood-brain barrier disruption | Important evidence for enhancing tumor, combined with the other channels |
| T2-weighted | T2, T2w | Fluid/water-rich tissue commonly appears bright; helps reveal abnormalities | One input channel; useful for inspecting lesion extent |
| T2-FLAIR | FLAIR, t2f | Fluid-attenuated inversion recovery: suppresses cerebrospinal-fluid signal while retaining many T2-related tissue abnormalities | Useful for inspecting surrounding abnormal signal; default viewer background |

Our BraTS workflow will combine all four contrasts, using **T1Gd/T1ce** for post-gadolinium T1 imaging. Reports will describe enhancement and FLAIR abnormalities as imaging findings, with histological interpretation outside the prototype's scope. [Official BraTS modality description](https://www.med.upenn.edu/cbica/brats2020/data.html), [MRI protocol reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC10466217/pdf/nihms-1919839.pdf)

We will calculate physical volumes from segmentation masks using their spatial metadata, and support viewing across all four modalities. The case manifest will retain acquisition spacing so the viewer and report reflect the source resolution, including when a volume was reconstructed from 2D slices. Resampled views will retain that provenance.

We will accept **NIfTI** files (`.nii` and gzip-compressed `.nii.gz`) and store modality assignments separately from file format. Each image's affine will map zero-based voxel coordinates `(i,j,k)` into reference coordinates `(x,y,z)`. We will check spatial units and convert to mm for measurements:

```text
[x,y,z,1]ᵀ = A[4×4] × [i,j,k,1]ᵀ
voxel_volume_mm3 = abs(det(A[:3,:3]))  # affine converted to mm
region_volume_ml = region_voxel_count × voxel_volume_mm3 / 1000
```

We will validate header geometry and inspect alignment before inference. Crosshairs will use the image's world coordinate system; standardized atlas coordinates will require a future atlas-registration step. [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html)

| Region | Definition in our older BraTS label scheme | Derived mask |
|---|---|---|
| Enhancing tumor (ET) | Enhancing tumor region | `L == 4` |
| Tumor core (TC) | Enhancing plus necrotic/non-enhancing core | `(L == 1) OR (L == 4)` |
| Whole tumor (WT) | Core plus the surrounding edema/infiltrative-abnormality region represented by the benchmark | `L in {1,2,4}` |

We will report these nested regions separately using `ET ⊆ TC ⊆ WT`, with WT providing the overall benchmark-region volume. The multimodal model will predict all three from the four inputs together, and the interface will support cases with an empty enhancing region. A future postoperative workflow will introduce a separate label mapping for resection cavities. [MONAI region definitions](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md)

### How incoming scans are categorized

We will categorize **each series by modality**, then assess **the case's model eligibility** using the declared clinical context and input requirements.

1. For bundled datasets, read their manifest and documented suffixes. Map `t1→t1`, `t1ce/t1gd/t1c→t1ce`, `t2/t2w→t2`, and `flair/t2f→flair`; map `t1n→t1` for a dataset that defines it that way. Match full suffix tokens, testing T1ce before T1 so it is not misclassified.
2. For user uploads, provide four named slots plus an `unknown` viewer slot. Filename matching will suggest assignments for user confirmation. Ambiguous filenames will remain in the unknown slot until the modality and contrast status are confirmed.
3. Use dataset manifests or sidecar metadata when present. NIfTI headers alone often cannot reliably distinguish these contrasts. DICOM metadata-based classification/conversion is a future adapter, not part of this NIfTI MVP.
4. Require one selected series per modality, all from the same subject and examination. Never combine different patients or pre/post-treatment dates to fill missing channels. Verify manifest/session identity or obtain an explicit user declaration; geometry alone cannot establish identity.
5. Record declared `population=adult|pediatric|unknown`, `treatment_state=preoperative|postoperative|unknown`, and `target_condition=glioma|other|unknown`. Official sample manifests populate these values. Unknown or unsupported clinical context permits viewing but blocks automated segmentation in this prototype.
6. Check geometric compatibility, finite intensities, prepared 1 mm spacing, skull-stripping/co-registration provenance and supported file dimensions. Identical shape/affine is necessary for our profile but does not prove anatomical alignment.

No automatic MRI-sequence classifier is required for v1. Raidionics lists a sequence-classification model but explicitly describes it as unvalidated; do not use it to silently route inference. [Raidionics model inventory](https://github.com/raidionics/Raidionics-models)

### Model routing and requirements by incoming scan combination

**We will use one multimodal segmentation model with four MRI input channels to produce the three tumor-region outputs.**

| Incoming case | Selected behavior/model | Requirements or reason |
|---|---|---|
| Complete T1 + T1ce + T2 + FLAIR, prepared preoperative adult glioma case | Run MONAI `brats_mri_segmentation` SegResNet | Same examination, supported geometry, four confirmed modalities; model order `[T1ce,T1,T2,FLAIR]` |
| Same complete input, research comparison requested | Swin UNETR BraTS21, future option | Separate pinned checkpoint and its own preprocessing/channel-order adapter; not an automatic fallback |
| Only T1, T1ce, T2 or FLAIR | View the available volume; segmentation disabled | Selected SegResNet requires all four channels |
| Any two or three of the required modalities | View available volumes; display exact missing modalities | Do not copy channels, substitute FLAIR for T2, or fill missing channels with zeros |
| Four files but unknown/duplicate modality assignment | View each file separately until resolved | Four files do not necessarily equal four required contrasts |
| Complete raw/unregistered or non-skull-stripped scans | View; block model execution | Needs a validated preprocessing pipeline; spacing changes alone do not register anatomy |
| Postoperative glioma, even with all four sequences | View; no automatic inference with the v1 model | Resection cavities and treatment effects are outside this prototype's supported task |
| Pediatric tumor, metastasis, meningioma or unknown condition | View; no automatic inference with the v1 model | Requires an appropriate task-specific checkpoint and validation |
| CT, DWI/ADC, perfusion, PET, 2D JPEG/PNG, ambiguous 4D data | Outside v1 segmentation support | These are not substitutes for the four-channel BraTS inputs |

For a **future postoperative branch**, Raidionics documents `MRI_GBM_Postop_FV_1p` for postoperative T1ce, `FV_2p` for postoperative T1ce+T1, and `FV_3p` for postoperative T1ce+T1+FLAIR. These are candidates for that specific task, not general missing-modality substitutes for our preoperative SegResNet. Their outputs must not be assumed to reproduce WT/TC/ET; inspect checkpoint configuration and validate separately before enabling them. [Released model descriptions](https://github.com/raidionics/Raidionics-models)

HD-GLIO also requires all four modalities, so it does not solve missing input scans. Its output definitions differ from the baseline. A model trained for missing modalities would be a separate future research track; withholding a channel from our existing model is not validation of that capability. [HD-GLIO requirements](https://github.com/CCI-Bonn/HD-GLIO)

Runtime requirement for the selected path: a CUDA GPU worker running PyTorch/MONAI, CPU RAM for the decoded four-channel case and output buffers, disk space for case artifacts, and a browser supported by the pinned NiiVue release. Target a 16 GB VRAM cloud GPU for development; this is a practical starting allocation, not a verified minimum or speed guarantee. Measure actual usage in the first inference milestone. The browser only renders; it does not execute this model.

### Dataset availability for each modality and task

We will source cases from **paired multimodal collections**, obtaining all four sequences from the same subject and examination. This keeps modality inputs and reference masks aligned throughout development and evaluation.

| Dataset | T1 | T1ce/T1Gd | T2 | FLAIR | Masks and intended use |
|---|---|---|---|---|---|
| [BraTS 2018](https://www.med.upenn.edu/sbia/brats2018.html) | Yes | Yes | Yes | Yes | Labelled training portion; reproduction/compatibility, with known risk of overlap with selected model training |
| [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) | Yes | Yes | Yes | Yes | Labelled training portion; development and disclosed evaluation; user's Kaggle mirror remains an access option |
| [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) | Yes | Yes | Yes | Yes | Preferred demo source; public training masks, challenge validation/test labels may be hidden |
| [MSD Task01](https://medicaldecathlon.com/) / [AWS](https://registry.opendata.aws/msd/) | Yes | Yes | Yes | Yes | Labelled training portion; modalities packed in 4D NIfTI; use dataset.json for ordering and label conversion |
| [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) | Yes | Yes | Yes | Yes | Preoperative diffuse-glioma collection with segmentations and additional modalities; select the prepared structural data and inspect release manifest |
| [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | Available | Available | Available | Available | Prepared MRI and corrected masks; check per-subject availability and overlap before selecting cases |
| [BraTS Lighthouse 2025](https://www.synapse.org/Synapse%3Asyn64153130/) | Yes | Yes | Yes | Yes | Pre/post-treatment task; different labels including cavity; future model route, access conditions apply |

“Yes” describes the collection's intended sequence set, not a guarantee that every mirror/file has been downloaded and checked. Save the exact release and per-case inventory. UCSF-PDGM's original paper describes 500 subjects, while the current TCIA description lists 501; do not mix release counts. [Current collection record](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)

All four sequences can be used independently for viewer testing. Full segmentation testing needs a complete compatible case; Dice additionally needs a reference mask. Curated incomplete cases can test the missing-modality UI but cannot establish segmentation accuracy. Do not call a cohort independent until patient overlap has been checked.

### Eligibility fields and API/UI changes

Extend `POST /api/cases` to accept one or more named modality files for viewing, plus an optional `unknown` file. Require at least one readable 3D NIfTI. Inference still requires exactly one selected input for each of the four known modalities. For incomplete/mismatched cases, do not overlay unaligned volumes as if registered; allow individual viewing.

Add multipart fields `population`, `treatment_state`, `target_condition` (enums above, default `unknown`), `same_exam_confirmed` and `prepared_inputs` (Boolean, default false). A trusted curated sample manifest supplies them automatically; user uploads require confirmation. These are provenance assertions, not automated medical classifications.

Return a structured eligibility object alongside the case ID:

```json
{
  "available_modalities": ["t1", "t2", "flair"],
  "missing_modalities": ["t1ce"],
  "modality_assignment_status": "confirmed",
  "eligibility": {
    "status": "view_only",
    "model_id": null,
    "reason_codes": ["MISSING_MODALITY"],
    "message": "Add T1 post-contrast from the same examination to enable segmentation."
  }
}
```

Allowed assignment states: `suggested|confirmed|ambiguous`. Eligibility states: `ready|view_only`; invalid files receive a validation error rather than a saved eligible case. Reason codes include `MISSING_MODALITY`, `AMBIGUOUS_MODALITY`, `DUPLICATE_MODALITY`, `GEOMETRY_MISMATCH`, `PREPROCESSING_REQUIRED`, `EXAM_IDENTITY_UNCONFIRMED`, `UNSUPPORTED_POPULATION`, `UNSUPPORTED_TREATMENT_STATE`, and `UNSUPPORTED_CONDITION`.

Recheck eligibility on `POST /api/cases/{id}/jobs`; return HTTP 422 with reasons if not ready. For v1, the server selects the sole validated model; a client cannot bypass routing by providing a model name. To correct a partial case in the MVP, submit a new complete case rather than mutating an existing case's input files.

Add acceptance cases: complete eligible case runs; T1-only and three-modality cases remain viewable but cannot run; `_t1ce` never becomes T1; unknown suffixes require confirmation; duplicate T1 does not satisfy T1ce; different-session files are not combined; postoperative/pediatric/unknown-context cases cannot run the baseline; empty ET is a valid prediction, not a routing failure.

### User flow

Choose sample or upload available scans → confirm modality assignments and case context → validate inputs and show eligibility → view available scans → if eligible, queue inference → review axial/coronal/sagittal/3D views → inspect regional volumes → optionally correct mask → export.

Use one NiiVue multiplanar canvas with its render panel, modality switching, crosshairs, orientation labels, window/level, region visibility and overlay opacity. Use named API constants from the pinned package instead of hard-coded slice-type numbers. [Layouts](https://niivue.com/docs/layouts/) and [drawing tools](https://niivue.com/docs/drawing/) support this approach.

### Backend behavior

- FastAPI owns cases, an in-memory job queue and per-job files; one persistent GPU worker loads the model once and processes jobs serially. Return a job ID immediately; poll every two seconds. Restarted in-progress jobs become failed/retryable.
- Validate readable NIfTI data, finite intensities, non-singular affine and spatial units for viewing; additionally require four confirmed modalities, supported case context and compatible shape/affine before inference. Geometry compatibility does not prove anatomical co-registration; require prepared inputs and visually inspect samples.
- Use the documented channel order and nonzero per-channel intensity normalization. Preserve the model's prepared coordinate convention. Do not blindly add RAS reorientation simply because another tutorial uses it. For v1, reject inputs outside the tested prepared geometry with a clear message; arbitrary raw-scan registration/skull stripping is deferred.
- Start with the bundle's inference settings: 240×240×160 sliding window, overlap 0.5, batch 1, GPU mixed precision. Measure memory and latency. If it runs out of memory, test a 128×128×128 window with the same overlap, record the change and recheck outputs before freezing configuration.
- Threshold sigmoid outputs at 0.5, use the bundle's ET→4, otherwise TC→1, otherwise WT→2 conversion, and derive displayed nested regions from the final label mask. Preserve original prediction separately from edits. [Inference config](https://huggingface.co/MONAI/brats_mri_segmentation/raw/main/configs/inference.json)
- Export the label mask aligned with the accepted input reference. If spatial preprocessing is introduced later, invert recorded transforms, use nearest-neighbor interpolation for discrete labels, and verify the output grid.
- Compute volume as N × abs(det(affine[:3,:3])) / 1000 after converting spatial units to millimetres. TC, WT and ET overlap; never add their volumes into a total. Report measurement of a predicted mask, not an exact biological tumor volume.

### Minimal interfaces

| Endpoint | Purpose |
|---|---|
| POST /api/cases | Available modality files plus case-context fields; return case ID, validation and inference eligibility |
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
