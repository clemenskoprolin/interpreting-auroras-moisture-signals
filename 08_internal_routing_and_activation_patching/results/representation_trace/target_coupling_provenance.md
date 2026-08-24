# Target-coupling aggregate provenance

This file records the compact aggregate rows used by
`visualize/fig_target_coupling_bars.py` and the paper's target-coupling panel.

The values are local median signed hotspot responses to matched one-standard-
deviation input perturbations. Raw output units are shown here; the paper
figure divides each value by the corresponding climatological output scale.

| Perturbed input | Output | Precipitation + ZWD model | Precipitation-only model |
|---|---|---:|---:|
| q850 | q850 (g/kg) | +0.918 | +0.967 |
| q850 | precipitation (mm/6 h) | +0.265 | +0.313 |
| q850 | ZWD (mm) | +0.659 | structurally absent |
| precipitation | q850 (g/kg) | +0.0002 | +0.0041 |
| precipitation | precipitation (mm/6 h) | +0.316 | +0.346 |
| precipitation | ZWD (mm) | -0.198 | structurally absent |
| ZWD | q850 (g/kg) | +0.143 | structurally absent |
| ZWD | precipitation (mm/6 h) | +0.218 | structurally absent |
| ZWD | ZWD (mm) | +52.0 | structurally absent |

The output scales are 4.075226373970509 g/kg for q850, 2.11 mm/6 h for
precipitation, and 98.5413 mm for ZWD. The q850-input rows come from
`q850_trace_all_pair_runs.csv` and `q850_self_trace_all_pair_runs.csv`; the
precipitation-only rows come from `precip_only_coupling_all_pair_runs.csv`.
The ZWD and precipitation perturbation rows are the aggregate values recorded
by the completed representation trace. The plotting script encodes the table
directly and applies the stated output scales for its standardized view.
