# Internal routing and activation patching

Two final analyses are retained here.

1. The 22-case representation trace applies matched one-standard-deviation
   perturbations to ZWD, 850 hPa humidity, and precipitation at saliency
   hotspots and matched low-near controls. It records output coupling,
   stage-wise hidden-state displacement, factorial ZWD--precipitation
   interaction, and cross-model CKA.
2. Unified activation patching tests ZWD, humidity, and precipitation in the
   same precipitation+ZWD checkpoint, on the same eight cases and 850 hPa
   humidity target. Source activations are copied into spatial-mean-baseline
   runs at individual U-Net sites and regions.

The exact cohorts are fixed by
`../07_zwd_precipitation_model_comparison/cases_diagnostic_22.json` and
`cases_activation_patch_8.json`. The trace consumes the Stage-B saliency maps
from directory 07 to define hotspot and matched-control regions.

## Representation trace

Run in a GPU allocation after Stage B has produced its per-case maps:

```bash
python 08_internal_routing_and_activation_patching/trace_precip_representations.py \
  --model-conditional-dir results/model_conditional_comparison \
  --diagnostic-selection 07_zwd_precipitation_model_comparison/cases_diagnostic_22.json \
  --modes zwd_trace precip_trace q850_trace factorial cka \
  --targets q850 precip zwd \
  --scales local synoptic \
  --output-dir results/internal_routing
```

The thesis uses the trace to show that low-near controls can cause equal or
larger RMS hidden-state changes while producing far smaller output responses,
and that cross-model CKA remains approximately 0.989--0.999. Hidden-state
magnitude alone is therefore not an importance measure.

The compact retained evidence is under `results/representation_trace/`:
portable JSON run configurations, all-pair and stage-metric CSV aggregates,
and `target_coupling_provenance.md`. These files reproduce the target-coupling
evidence used in the paper.

## Unified activation patching

Run each input with the same manifest, sites, regions, and leads:

```bash
python 08_internal_routing_and_activation_patching/activation_patch_all_inputs.py \
  --variable zwd
python 08_internal_routing_and_activation_patching/activation_patch_all_inputs.py \
  --variable q
python 08_internal_routing_and_activation_patching/activation_patch_all_inputs.py \
  --variable precip
```

The common default output root is
`$AURORA_XAI_RESULTS_DIR/activation_patch_all_inputs/`. The retained figure
scripts summarize target coupling and unified patch recovery from saved CSVs.
The completed eight-case shard configurations and score tables are versioned
under `results/activation_patch_all_inputs/`; the figure script reads that
directory by default. Its JSON paths are portable substitutions for the
original cluster paths, while measured values and case splits are unchanged.
The two unit-test files validate mask downsampling, normalization, and
aggregation without importing Aurora.

Reproduce the paper's standardized coupling panel from the repository root
with:

```bash
python 08_internal_routing_and_activation_patching/visualize/fig_target_coupling_bars.py \
  --standardized \
  --output figures/target_coupling_bars.png
```
