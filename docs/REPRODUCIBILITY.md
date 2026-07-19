# Reproducibility guide

## Environment

Use MATLAB R2024a and record the output of `ver`. Run without parallel workers unless a future revision explicitly controls parallel random streams. The scripts set fixed Twister seeds for repeatability.

## Recommended run order

1. Validate the private input files against `DATA_CONTRACT.md`.
2. Run `overlap_fuse_spectra` on a small subset and inspect the returned `meta` table.
3. Run `train_validate_modern_classifier` for the primary analysis.
4. Run `compare_overlap_aware_classifiers` on the identical split.
5. Run `finalize_vip_feature_attribution` for interpretation tables.
6. Run the two `probe_*` functions only as separately labeled robustness analyses.

## Review checklist

- Confirm that the final-test rows never participate in tuning.
- Confirm that target-batch labels are absent from alignment/model fitting.
- Compare saved split assignments and random seeds between runs.
- Inspect confusion matrices for minority-class failure hidden by accuracy.
- Treat emission assignments as hypotheses requiring spectroscopy review.
- Record MATLAB/toolbox versions and Git commit ID with every reported result.

## Expected outputs

The scripts create CSV/MAT result artifacts under `reports/` and figures under `figures/`. Exact names are defined in each entry-point function. Generated outputs should normally remain out of Git unless a small, reviewed table is intentionally included for a release.

## Known limitations

- The sample size is small and class counts are imbalanced.
- Labels and channel counts are currently tailored to the study dataset.
- Some exploratory scripts use compact local helper functions and need additional unit tests.
- VIP values describe model-associated importance; they do not by themselves prove a mechanistic causal relationship.
