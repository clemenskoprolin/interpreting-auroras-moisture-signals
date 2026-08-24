# XAI verification suite (paper appendix)

Self-contained diagnostics that reproduce the numbers in the XAI-methods
verification appendix. Method settings are pinned in `_suite_common.py` so the
results do not depend on the engine's current shipped defaults.

## Contents

| script | produces | appendix table |
|--------|----------|----------------|
| `completeness_suite.py` | IG completeness across 9 cases | IG completeness |
| `randomization.py` | cascading + full-model randomization, spatial baselines | full-model randomization |
| `rise_convergence.py` | RISE self-consistency / reproducibility vs mask count | RISE mask-count convergence |
| `_suite_common.py` | shared case grid, metrics, pinned settings | — |
| `aggregate.py` | merge per-rank JSONL → Markdown tables | — |
| `out/*_summary.md` | reference results | — |

## Pinned settings (`_suite_common.py`)

- RISE: `RISE_MASKS` masks on an 18×36 low-resolution mask grid, p=0.5
- ViT-CX: encoder stage 2 (token grid 4×45×90), 256 clusters
- IG: σ=2.5° smoothed baseline (reflect lat, wrap lon), midpoint Riemann, steps {4,8,16,32,64}
- Randomization seeds {11,23,37,51,67}; cascade seeds {11,23}
- RISE convergence: seeds A=42 / B=1234; checkpoints {32,64,128,256,512,1024}

Each suite shards across SLURM ranks and streams per-rank JSONL into `out/`,
which `aggregate.py` merges into the summary tables. The submitted cases and
methods are configurable per run via environment variables (e.g. `RISE_CASES`,
`SF_SANITY_CASES`, `DIAG_METHODS`, `RISE_CHECKPOINTS`).
