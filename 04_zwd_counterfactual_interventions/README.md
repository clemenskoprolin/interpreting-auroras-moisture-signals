# ZWD counterfactual interventions

This experiment asks whether ZWD affects Aurora beyond the column-moisture
state already represented in ERA5. At each of the two input times, a global
ordinary least-squares relation reconstructs ZWD from native ERA5 total column
water vapour (TCWV). Aurora is then evaluated with true ZWD and with this
TCWV-derived surrogate while every other input remains fixed.

The retained cohort has 228 cases over 19 targets: 12 cases per target, balanced
across low, typical, humid, and extreme monthly-TCWV strata. The output is the
change in predicted 850 hPa specific-humidity box mean at 6, 12, and 24 hours.

The versioned aggregate table in
`results/stratified_decomposition_combined228.csv` is the source for the final
thesis result. Mean true-minus-surrogate responses were -0.0218, -0.0294, and
-0.0342 g/kg over mountain targets and -0.0081, -0.0102, and -0.0159 g/kg over
non-mountain targets at 6, 12, and 24 hours in the thesis table. A direct mean
of the retained floating-point rows gives -0.01585 g/kg at 24 hours, which
rounds to -0.0158 under round-to-even; the thesis displays -0.0159. The versioned
`results/tcwv_residuals_228cases.csv` records the matching input residuals.
These are aggregate research outputs, not a forecast-skill evaluation.
The two versioned CSV tables are the compact numerical outputs, while the
cohort definitions and inputs are JSON manifests.

## Cohorts

The three manifests and matching metadata partition the final cohort:

- `cases_stratified96.json`: 96 cases across the first eight targets.
- `cases_stratified_extension84.json`: 84 cases across seven additional targets.
- `cases_stratified_flat_extension48.json`: 48 cases across four flat controls.

`select_stratified_cases.py` documents and reproduces the data-only stratified
selection. `check_iwv_baseline_variants.py` is the supporting diagnostic that
motivated use of native TCWV instead of a pressure-level integral that includes
below-ground levels over mountains.

## Run

Run each manifest in a GPU allocation with only the thesis replacement modes:

```bash
python 04_zwd_counterfactual_interventions/conditional_zwd_mechanism.py \
  --cases 04_zwd_counterfactual_interventions/cases_stratified96.json \
  --leads 6 12 24 \
  --modes true qhat_tcwv \
  --no-stage-b --no-stage-c --no-stage-d \
  --output-dir results/zwd_counterfactual_interventions/part96
```

Repeat for the 84- and 48-case manifests with distinct output directories.
Then combine the three `ablation_scores.csv` files:

```bash
python 04_zwd_counterfactual_interventions/analyze_stratified.py \
  --results results/zwd_counterfactual_interventions/part96/ablation_scores.csv \
            results/zwd_counterfactual_interventions/part84/ablation_scores.csv \
            results/zwd_counterfactual_interventions/part48/ablation_scores.csv
```

`compute_matching_tcwv_residuals.py` recomputes the input-side residual table
from the combined case table without importing Aurora.
