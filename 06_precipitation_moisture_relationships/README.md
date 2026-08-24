# Precipitation--moisture relationships

This CPU/data-only experiment repeats the regional diagnostic with MSWEP
precipitation as the outcome. It uses Ticino, California, and Japan throughout
2020, giving 2,193 pooled target-time samples.

The lag convention is explicit: positive `lag_hours` pairs moisture at time
`t` with precipitation at `t + lag`. The final analysis evaluates 0, 6, 12,
18, and 24 hours. Six- and 18-hour outcomes require the supplemental 06/18 UTC
precipitation time series added to the final code.

At lag zero, the thesis reports a strongest pooled Spearman correlation of
0.422 for box-mean precipitation versus box-mean ZWD. Across feature pairs,
mean absolute Spearman correlation is 0.328 for ZWD and 0.260 for humidity.
The association rises slightly at six hours and decays to roughly 0.17--0.21
at 24 hours.

`results/top_correlations.json` is the completed run's machine-readable ranked
summary. The pipeline also produces the paired time series and full correlation
matrix used by the retained plotting script.

## Run

```bash
python 06_precipitation_moisture_relationships/run_all.py \
  --start 2020-01-01T00:00:00 \
  --end 2020-12-31T00:00:00 \
  --candidate-hours 0 12 \
  --lag-hours 0 6 12 18 24 \
  --targets ticino california japan \
  --regions box disk \
  --output-dir results/p_correlation_diagnostics
```

`run_all.py` builds the paired time series, adds the supplemental precipitation
timestamps, and computes raw and monthly-standardized correlations. The
retained plotting script reads
`correlations/correlations.csv` and produces the regional and lag summary.
No Aurora checkpoint is loaded.
