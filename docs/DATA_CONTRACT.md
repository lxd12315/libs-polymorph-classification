# Data contract

The scripts expect local files under `data/classification/`. These files are intentionally excluded from the repository.

## `julei.mat`

Required variables:

- `spec_al`: numeric matrix with at least 12,286 wavelength channels and 37 columns. Columns are samples; rows are spectral channels.
- `spec_al2`: independent robustness batch with the same channel and sample ordering.
- `significant_peaksCopy`: indices of selected spectral variables, required by VIP attribution.

Current class ordering is fixed in the research scripts:

- columns 1–24: polymorph A
- columns 25–37: polymorph B

If a different dataset is used, replace this hard-coded label construction with explicit metadata and add validation assertions.

## `w.mat`

- `w`: wavelength vector aligned with the spectral rows. Repeated/non-increasing boundaries identify separate detector segments.

## `reports/fixed_sample_split.csv`

Required logical or binary columns:

- `Modeling_set`
- `Final_test_set`

Each labeled sample must belong to exactly one subset. The final-test assignment must be fixed before tuning.

## Validation recommendations

Before running a model, verify finite numeric values, wavelength/spectrum dimension agreement, mutually exclusive split membership, both classes in the development subset, and consistent channel order across the main and robustness batches.
