# Method

## Objective

The workflow distinguishes two anonymized polymorph classes from LIBS spectra while reducing optimistic bias caused by preprocessing or model selection on evaluation samples. The material identity is intentionally outside the scope of this public repository.

## Processing sequence

1. Load the wavelength vector and sample-by-channel spectra.
2. Detect detector-channel boundaries where wavelength values cease increasing.
3. Interpolate each detector segment to a shared grid and blend overlapping segments with complementary raised-cosine weights.
4. Apply sample-wise spectral normalization where required by the selected experiment.
5. Use the predefined stratified development/final-test assignment.
6. Estimate every preprocessing parameter from development data only.
7. Tune the classifier using repeated stratified cross-validation inside the development subset.
8. Fit the selected configuration on all development samples and evaluate the untouched final-test subset once.
9. Apply the locked pipeline to the independent robustness batch.

## Models

`train_validate_modern_classifier.m` implements a leakage-aware PCA/RBF-SVM workflow. PCA dimension and SVM hyperparameters are selected only inside the development set.

`compare_overlap_aware_classifiers.m` compares candidate classifiers on identical fixed splits. It includes a proximity-style method whose representation and neighbor/prototype decisions are learned only from training folds.

`probe_multirepresentation_svm.m` tests raw spectra, first derivatives, second derivatives, and concatenated representations. It is an exploratory robustness probe and should not be reported as confirmatory if the final test results influenced model selection.

`probe_coral_domain_alignment.m` evaluates unsupervised covariance alignment between the development domain and an independent target batch. Target labels are used for evaluation only, never for tuning.

## Metrics

- Balanced accuracy: mean recall across the two polymorph classes.
- Macro-F1: unweighted mean of class-wise F1 scores.
- Sensitivity and specificity: report with an explicit positive-class definition.
- Confusion matrix: retain sample counts, not only percentages.
- Prediction margin/confidence: use for uncertainty description and possible open-set flags, not as proof of calibration.

## Feature interpretation

`finalize_vip_feature_attribution.m` computes OPLS/PLS-derived VIP scores, maps selected wavelengths to candidate atomic or molecular emissions, compares class-wise mean intensity, and aggregates weights by assignment. Spectral assignment requires domain review because wavelength proximity alone cannot establish chemical identity.

## Leakage controls

- Never fit PCA, scaling, feature selection, or thresholds using final-test samples.
- Never choose a random seed because it produces favorable final-test accuracy.
- Preserve the split table with the results.
- Keep exploratory probes distinct from the primary confirmatory analysis.
- Treat independent-batch results as external robustness evidence, not as another tuning set.
