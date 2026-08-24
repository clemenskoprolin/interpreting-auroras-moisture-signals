# Attribution method validation

This directory contains the validation experiments reported for the
ZWD-augmented Aurora checkpoint. The experiments use 6 h forecasts.

## Experiments

- `stability.py` adds Gaussian noise equal to 5% of the ZWD standard deviation
  and compares saliency and Integrated Gradients (IG) maps over three seeds.
- `randomization.py` cumulatively randomizes decoder and encoder stages and
  measures how attribution similarity to the trained model collapses.
- `completeness.py` evaluates the IG completeness residual over 8, 16, 32, and
  64 integration steps.
- `rise_convergence.py` measures within-run convergence and between-seed
  reproducibility for RISE over 32 to 1024 masks on the three strong cases.
- `completeness_humidity_finetuned.py` is the supporting comparison that applies
  the same completeness test to specific humidity in the fine-tuned ZWD and
  precipitation checkpoints.
- `diagnostics.py` provides one command-line entry point for the first four
  experiments. `common.py` contains their shared configuration and utilities.

## Results reported in the thesis

- Mean stability Pearson correlation was 0.998 for saliency and 0.979 for IG;
  top-10 mask overlap was 1.0 for both methods.
- After randomizing the outer decoder stage, Pearson correlation was 0.69 for
  saliency and 0.66 for IG. It fell to approximately zero after the complete
  encoder cascade (−0.002 and 0.044, respectively).
- The 6 h IG completeness residual stabilized at 13.2% at 64 steps.
- At 1024 masks, between-seed RISE reproducibility was 0.53 for Ticino, 0.57 for
  Japan, and 0.64 for California.

## Versioned outputs

`results/` contains the machine-readable JSON for 6 h completeness,
cumulative randomization, stability, the three-case RISE runs, and native-
humidity completeness controls. Public-Aurora verification summary tables are
under `../geoxplain-aurora-adapter/diagnostics/out/`.

## Running

These scripts initialize Aurora and therefore must run inside a GPU allocation,
not on a login node. From the repository root, for example:

```bash
python 01_attribution_method_validation/diagnostics.py stability
python 01_attribution_method_validation/diagnostics.py randomization
python 01_attribution_method_validation/diagnostics.py completeness
python 01_attribution_method_validation/rise_convergence.py
```

Each experiment writes machine-readable JSON beneath
`$AURORA_XAI_RESULTS_DIR/attribution_method_validation/`. If the environment
variable is not set, `results/` in the repository root is used. The output root
can also be overridden directly with `ATTRIBUTION_VALIDATION_RESULTS_DIR`.

Stability and randomization read their reference maps from
`$ZWD_SEARCHLIGHT_RESULTS_DIR`. By default this resolves to the fixed-regime 6 h
point benchmark under the shared results root.
