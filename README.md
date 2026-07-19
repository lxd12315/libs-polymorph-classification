# LIBS Polymorph Classification

MATLAB R2024a workflow for leakage-aware classification of different material polymorphs from LIBS spectra. The repository contains the reusable algorithmic core only; material identities, unpublished spectra, manuscripts, and personally identifying paths are intentionally excluded.

## Highlights

- Detector-overlap correction with raised-cosine fusion on a common wavelength grid.
- Fixed development/final-test split; the final test set is isolated from model selection.
- Repeated stratified cross-validation for small-sample model selection.
- PCA + class-balanced RBF-SVM baseline and proximity-style prototype comparison.
- Balanced accuracy, macro-F1, sensitivity, specificity, confusion matrices, and prediction margins.
- Independent-batch robustness probes, multi-representation spectra, and diagnostic CORAL alignment.
- VIP-based spectral feature attribution with emission assignments.

## Repository layout

```text
src/       MATLAB functions
docs/      method, data contract, and reproducibility notes
examples/  local configuration template
reports/   generated result tables (kept empty in source control)
```

## Requirements

- MATLAB R2024a
- Statistics and Machine Learning Toolbox
- Data files following [the data contract](docs/DATA_CONTRACT.md)

## Quick start

Place local, non-public input files under `data/classification/` (this directory is ignored by Git), then run:

```matlab
projectRoot = pwd;
addpath(fullfile(projectRoot, 'src'));

train_validate_modern_classifier(projectRoot);
compare_overlap_aware_classifiers(projectRoot);
finalize_vip_feature_attribution(projectRoot);
```

Optional robustness diagnostics:

```matlab
probe_multirepresentation_svm(projectRoot);
probe_coral_domain_alignment(projectRoot);
```

See [METHOD.md](docs/METHOD.md) for the experimental design and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a full run checklist.

## Important interpretation limits

The current labels encode two anonymized polymorph classes, A and B. A separate unknown sample, when present, must not be treated as a supervised third class. Accuracy from this small dataset should be reported together with class-balanced metrics and uncertainty; independent-batch performance is the stronger measure of robustness.

## Data availability

Raw spectra are not included because they are unpublished research data. Researchers can use the documented variable names and dimensions to connect authorized data without modifying the algorithm.

## License and citation

No license has been selected yet. Until the repository owner adds one, copyright remains reserved and reuse requires permission. Add the associated paper citation and DOI after publication.
