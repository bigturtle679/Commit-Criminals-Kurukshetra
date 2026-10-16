# Labels and regions

Resolve label meaning from dataset documentation and confirm observed values. Unique values
alone cannot establish anatomy. Store both semantic mapping and supporting source per case.
The existing real GLI demo masks use ET=3; the existing synthetic mask uses ET=4.
New Kaggle cases need their own verified mapping.

Whole tumor (WT) includes nonenhancing core, edema and enhancing tumor.
Tumor core (TC) includes nonenhancing core and enhancing tumor. ET is enhancing tumor.
Check checkpoint output region order separately from integer mask labels. The current
ensemble assumes output channels TC, WT, ET; verify that for both trained checkpoints.
Masks are evaluation-only and must never enter normalization, crop selection or model inputs.
