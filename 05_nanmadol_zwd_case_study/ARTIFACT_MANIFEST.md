# Artifact manifest

## Versioned inputs and code

- `data/nanmadol_ibtracs.csv`: curated IBTrACS v04r01 rows used by the figures.
- `storm_centered_xai.py`: baseline rollout, tracker, moving targets, saliency,
  and region selection.
- `track_perturbation.py`: hotspot and spatial-control perturbation rollouts.
- `plot_thesis_figure.py` and `plot_saliency_evolution.py`: plotting-only
  scripts for the two thesis figures.
- `verify_baseline_vs_besttrack.py`: saved-track verification.

## External or generated artifacts

The ZWD checkpoint, input datasets, per-lead saliency arrays, baseline and
perturbed forecast tracks, selection JSON, response CSVs, and figures are not
versioned. The expected result layout is documented in README.md and resolved
from `AURORA_XAI_RESULTS_DIR`.
