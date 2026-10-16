# Training-overlap audit — SwinUNETR checkpoint vs. pilot cohort

Date: 2026-10-17. Supersedes every `overlap_status: "unknown"` entry in
`outputs/testing/antigravity/manifest.json`, `smoke_manifest.json` and
`manifest_reinheart_brats2022.json`. Those manifests stay unchanged
because their SHA-256 hashes are recorded in the benchmark metadata.

## Verdict

**`known_training_overlap` for all 20 pilot cases, all 3 smoke cases
and the 3 local GLI demo cases.** The SwinUNETR Dice scores in
`benchmark_report.md` measure performance on data the model was trained
on. They are a pipeline sanity check, not held-out accuracy.

## Evidence

| Item | Finding |
|---|---|
| Checkpoint file | `models/swinunetr_brats/model_weights_BRATS-finetune.pt` |
| SHA-256 | `6fd4ec5dc690f365a99c10b92f0d5519ca8a6b68feb844ff3424e9f6a01f6a39` |
| Embedded metadata | `epoch=249`, `best_acc=0.9027` (no fold or split recorded) |
| Download source | Hugging Face [`smilelab/BrainSegFounder`](https://huggingface.co/smilelab/BrainSegFounder) (see `notes/RECOVERED_CHAT_SESSION_DI02.md`), not MONAI's own release |
| Model card | "SwinUNETR finetuning weights … BraTS tumor segmentation … 1,470 subjects". No fold, split or held-out ID list published |
| Paper | Cox et al., *BrainSegFounder* ([arXiv 2406.10395](https://arxiv.org/html/2406.10395v3)): Stage 2 self-supervised pretraining **and** Stage 3 fine-tuning both use the 1,251-subject BraTS 2021 Task 1 training set, with 5-fold splits identical to MONAI's SwinUNETR BraTS21 folds |
| Pilot cohort IDs | `BraTS2021_000xx`–`BraTS2021_01157`: all are BraTS 2021 Task 1 **training** IDs |
| Local demo cases | `BraTS-GLI-00000/00002/00003` are BraTS 2023 GLI renames of BraTS 2021 training cases → same overlap |

1,470 = 1,251 BraTS 2021 training + 219 validation subjects. Even if the
released weights came from a single fold, every pilot case was seen
either with labels (4 of 5 folds) or as unlabeled images during Stage 2
pretraining. Which fold produced these weights is not published, so no
case can be marked `excluded_from_known_training`.

## Consequences

- Do not call 0.905 / 0.924 / 0.876 Dice "state-of-the-art" or "accuracy"
  without stating that the cases were in the training data.
- The fair held-out reference is the published BraTS21 SwinUNETR 5-fold
  cross-validation (Hatamizadeh et al., 2022, [arXiv 2201.01266](https://arxiv.org/abs/2201.01266);
  fold-average Dice roughly WT 0.93, TC 0.92, ET 0.89). Those are
  source-reported figures, not our measurement; check the paper table before quoting.
- Runtime, VRAM, determinism and software-correctness results are unaffected.

## What would give a genuinely held-out number

None of the downloaded sources provide one. BraTS 2021/2022/2023-GLI
training cases all derive from the same 1,251 subjects, and the BraTS 2021
validation labels were never released. Candidate sources requiring separate
access and label-mapping verification:

1. BraTS-Africa (SSA) adult glioma cases: different population, labeled,
   not in BraTS 2021.
2. BraTS 2023+ GLI cases added after 2021, if their provenance confirms
   they are not BraTS 2021 subjects.
3. Your own team's trained checkpoint (e.g. ResEnc) with a recorded split,
   evaluated only on its held-out fold.

Until one of these is benchmarked, report no held-out accuracy figure.
