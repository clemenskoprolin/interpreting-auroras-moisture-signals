# ZWD--moisture relationships

This CPU/data-only analysis establishes the physical baseline used to
interpret ZWD attributions. It compares regional ZWD with pressure-level and
pressure-weighted humidity, temperature, wind, and surface proxies in Ticino,
California, and Japan from 2020 through 2024.

The final cohort contains 10,962 pooled target-time samples. The thesis reports
a monthly-standardized Spearman correlation of 0.988 between 1,000 km
disk-mean ZWD and pressure-weighted column humidity, and 0.921 between
disk-mean ZWD and 850 hPa humidity. The association is strongest in the lower
and middle troposphere and weakens aloft.

The completed run's ranked numerical summary is versioned as
`results/top_correlations.json`. The pipeline also produces the full time
series and correlation matrices used by the figure scripts.

## Run

```bash
python 03_zwd_moisture_relationships/run_all.py \
  --start 2020-01-01T00:00:00 \
  --end 2024-12-31T12:00:00 \
  --candidate-hours 0 12 \
  --targets ticino california japan \
  --regions box disk \
  --output-dir results/zwd_correlation_diagnostics
```

The pipeline writes per-target and combined time series, raw and
monthly-standardized correlations, summaries, and plots. The two retained
figure scripts reproduce the vertical-profile and ZWD--column-humidity views
from those saved tables. No Aurora checkpoint is loaded.
