# Training overlap and duplicates

BraTS release names and Kaggle dataset titles do not establish independent patients.
Retrieve available checkpoint training manifests and dataset provenance. Compare IDs,
source mappings and image contents across years and mirrors. File hashes detect identical
files; voxel-content hashes can identify re-encoded duplicates when geometry is equivalent.
Neither method proves independence after arbitrary preprocessing.

Classify each patient as known_training_overlap, excluded_from_known_training, or unknown.
The second category only applies to the training lists actually available, not all possible
pretraining. Report the scope of checks. Unknown overlap stays unknown. Publish exploratory
scores separately from any documented held-out subset. Never describe training cases as
independent test evidence.
