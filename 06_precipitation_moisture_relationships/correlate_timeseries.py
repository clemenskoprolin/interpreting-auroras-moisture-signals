#!/usr/bin/env python3
"""
Compute precipitation/proxy correlations from time-series CSVs.

The main distinction is between:
  - raw correlations: includes geography and seasonality
  - monthly_z correlations: within-target monthly standardized anomalies

Lag convention: lag_hours = 12 means proxy(t) is correlated with P(t+12h).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_DIR, ensure_dir, month_zscore, pairwise_corr, save_json, season_name


META_COLUMNS = {"target", "month", "year", "season", "level", "quantile"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "timeseries"))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--supplemental-precip-file",
        default=None,
        help=(
            "Optional precipitation-only CSV containing outcome timestamps not "
            "present in the paired input, for example 06/18 UTC values needed "
            "for 6 h and 18 h moisture leads."
        ),
    )
    p.add_argument("--driver-columns", nargs="+", default=None)
    p.add_argument("--lag-hours", type=int, nargs="+", default=[0, 12, 24])
    p.add_argument("--transforms", nargs="+", default=["raw", "monthly_z"],
                   choices=["raw", "monthly_z"])
    p.add_argument("--min-n", type=int, default=12)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def load_timeseries(input_dir: str) -> pd.DataFrame:
    combined = os.path.join(input_dir, "all_targets_timeseries.csv")
    if os.path.exists(combined):
        df = pd.read_csv(combined, parse_dates=["init_time"])
        return df.set_index("init_time").sort_index()

    frames = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith("_timeseries.csv") or name == "all_targets_timeseries.csv":
            continue
        path = os.path.join(input_dir, name)
        frames.append(pd.read_csv(path, parse_dates=["init_time"]).set_index("init_time"))
    if not frames:
        raise FileNotFoundError(f"No *_timeseries.csv files found in {input_dir}")
    return pd.concat(frames).sort_index()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def infer_driver_columns(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in ("p_box_mean", "p_disk_mean", "p_global_mean") if c in df.columns]
    rest = [c for c in numeric_columns(df) if c.startswith("p_") and c not in set(preferred)]
    return preferred + rest


def proxy_family(col: str) -> str:
    if col.startswith("zwd_"):
        return "zwd"
    if col.startswith("q_"):
        return "humidity"
    if col.startswith("p_"):
        return "precip_other"
    if col.startswith("t_") or col.startswith("2t_"):
        return "temperature"
    if col.startswith("wind_") or col.startswith("u_") or col.startswith("v_") or col.startswith("10"):
        return "wind"
    if col.startswith("msl_") or col.startswith("z_"):
        return "pressure_or_geopotential"
    return "other"


def transformed_frame(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    out = df.copy()
    cols = numeric_columns(df)
    if transform == "raw":
        return out
    if transform != "monthly_z":
        raise ValueError(f"Unknown transform: {transform!r}")

    if "target" not in out:
        for col in cols:
            out[col] = month_zscore(out[col])
        return out

    for col in cols:
        out[col] = out.groupby("target", group_keys=False, sort=False)[col].transform(month_zscore)
    return out


def lagged_pairs(
    df: pd.DataFrame,
    driver_col: str,
    proxy_col: str,
    lag_hours: int,
    *,
    driver_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    lag = pd.Timedelta(hours=lag_hours)
    pieces = []
    driver_source = df if driver_df is None else driver_df
    group_iter = df.groupby("target", sort=False) if "target" in df else [(None, df)]
    for target, sub in group_iter:
        if target is not None and "target" in driver_source:
            driver_sub = driver_source[driver_source["target"] == target]
        else:
            driver_sub = driver_source
        d = driver_sub[[driver_col]].copy()
        p = sub[[proxy_col]].copy()
        # Align the future precipitation value P(t + lag) with the moisture
        # proxy at t.  The joined index therefore denotes the proxy time.
        d.index = d.index - lag
        pair = d.join(p, how="inner")
        if "target" in sub:
            pair["target"] = target
        pieces.append(pair)
    if not pieces:
        return pd.DataFrame(columns=[driver_col, proxy_col])
    return pd.concat(pieces).sort_index()


def _row(
    *,
    target: str,
    transform: str,
    lag_hours: int,
    driver_col: str,
    proxy_col: str,
    pair: pd.DataFrame,
    min_n: int,
    season: str = "ALL",
) -> dict | None:
    work = pair[[driver_col, proxy_col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(work))
    if n < min_n:
        return None
    return {
        "target": target,
        "season": season,
        "transform": transform,
        "lag_hours": int(lag_hours),
        "driver_column": driver_col,
        "proxy_column": proxy_col,
        "proxy_family": proxy_family(proxy_col),
        "n": n,
        "pearson": pairwise_corr(work[driver_col], work[proxy_col], "pearson"),
        "spearman": pairwise_corr(work[driver_col], work[proxy_col], "spearman"),
    }


def compute_correlations(
    df: pd.DataFrame,
    *,
    driver_columns: list[str],
    transforms: list[str],
    lag_hours: list[int],
    min_n: int,
    supplemental_precip: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = numeric_columns(df)
    proxy_columns = [c for c in num_cols if c not in set(driver_columns)]
    rows = []
    seasonal_rows = []

    for transform in transforms:
        tdf = transformed_frame(df, transform)
        driver_frames = [tdf[["target", *driver_columns]]]
        if supplemental_precip is not None:
            supplemental_tdf = transformed_frame(supplemental_precip, transform)
            driver_frames.append(supplemental_tdf[["target", *driver_columns]])
        driver_tdf = pd.concat(driver_frames).sort_index()
        targets = sorted(tdf["target"].dropna().unique()) if "target" in tdf else ["ALL"]
        target_frames = [(target, tdf[tdf["target"] == target]) for target in targets]
        target_frames.append(("ALL", tdf))

        for target, sub in target_frames:
            driver_sub = driver_tdf if target == "ALL" else driver_tdf[driver_tdf["target"] == target]
            for driver_col in driver_columns:
                if driver_col not in sub:
                    continue
                for proxy_col in proxy_columns:
                    if proxy_col not in sub:
                        continue
                    for lag in lag_hours:
                        pair = lagged_pairs(
                            sub,
                            driver_col,
                            proxy_col,
                            lag,
                            driver_df=driver_sub,
                        )
                        row = _row(
                            target=target,
                            transform=transform,
                            lag_hours=lag,
                            driver_col=driver_col,
                            proxy_col=proxy_col,
                            pair=pair,
                            min_n=min_n,
                        )
                        if row is not None:
                            rows.append(row)

                        if row is not None:
                            pair = pair.copy()
                            pair["season"] = [season_name(m) for m in pd.DatetimeIndex(pair.index).month]
                            for season, season_pair in pair.groupby("season"):
                                srow = _row(
                                    target=target,
                                    season=season,
                                    transform=transform,
                                    lag_hours=lag,
                                    driver_col=driver_col,
                                    proxy_col=proxy_col,
                                    pair=season_pair,
                                    min_n=min_n,
                                )
                                if srow is not None:
                                    seasonal_rows.append(srow)

    corr = pd.DataFrame(rows)
    seasonal = pd.DataFrame(seasonal_rows)
    if not corr.empty:
        corr = corr.sort_values(
            ["transform", "lag_hours", "target", "driver_column", "proxy_family", "proxy_column"]
        )
    if not seasonal.empty:
        seasonal = seasonal.sort_values(
            ["transform", "lag_hours", "target", "season", "driver_column", "proxy_family", "proxy_column"]
        )
    return corr, seasonal


def write_summary(corr: pd.DataFrame, output_dir: str, top_k: int) -> None:
    if corr.empty:
        return

    summary_path = os.path.join(output_dir, "correlation_summary.md")
    lines = ["# Precipitation/Proxy Correlation Summary\n\n"]
    lines.append("Positive lag pairs the proxy at time t with `driver_column` at time t + lag.\n\n")

    for transform in ("monthly_z", "raw"):
        subset = corr[(corr["target"] == "ALL") & (corr["lag_hours"] == 0) & (corr["transform"] == transform)]
        if subset.empty:
            continue
        top = subset.reindex(subset["spearman"].abs().sort_values(ascending=False).index).head(top_k)
        lines.append(f"## Top {transform}, pooled targets, lag 0\n\n")
        lines.append("| driver | proxy | family | n | pearson | spearman |\n")
        lines.append("|--------|-------|--------|---:|--------:|---------:|\n")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['driver_column']} | {row['proxy_column']} | {row['proxy_family']} "
                f"| {int(row['n'])} | {row['pearson']:.3f} | {row['spearman']:.3f} |\n"
            )
        lines.append("\n")

    lines.append("## Family Means\n\n")
    lines.append("| transform | lag_h | target | family | mean_abs_spearman | max_abs_spearman |\n")
    lines.append("|-----------|------:|--------|--------|------------------:|-----------------:|\n")
    fam = (
        corr.assign(abs_spearman=corr["spearman"].abs())
        .groupby(["transform", "lag_hours", "target", "proxy_family"], dropna=False)["abs_spearman"]
        .agg(["mean", "max"])
        .reset_index()
        .sort_values(["transform", "lag_hours", "target", "mean"], ascending=[True, True, True, False])
    )
    for _, row in fam.iterrows():
        lines.append(
            f"| {row['transform']} | {int(row['lag_hours'])} | {row['target']} "
            f"| {row['proxy_family']} | {row['mean']:.3f} | {row['max']:.3f} |\n"
        )

    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {summary_path}")

    top_json = {}
    for transform in sorted(corr["transform"].unique()):
        subset = corr[(corr["target"] == "ALL") & (corr["lag_hours"] == 0) & (corr["transform"] == transform)]
        subset = subset.reindex(subset["spearman"].abs().sort_values(ascending=False).index).head(top_k)
        top_json[transform] = subset.to_dict(orient="records")
    save_json(top_json, os.path.join(output_dir, "top_correlations.json"))


def maybe_plot(corr: pd.DataFrame, output_dir: str, top_k: int) -> None:
    if corr.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots; matplotlib unavailable: {exc}")
        return

    plot_dir = ensure_dir(os.path.join(output_dir, "plots"))
    family_colors = {
        "zwd": "#756bb1",
        "humidity": "#2b8cbe",
        "precip_other": "#3182bd",
        "temperature": "#e34a33",
        "wind": "#31a354",
        "pressure_or_geopotential": "#636363",
    }
    for transform in ("monthly_z", "raw"):
        subset = corr[(corr["target"] == "ALL") & (corr["lag_hours"] == 0) & (corr["transform"] == transform)]
        if subset.empty:
            continue
        subset = subset.reindex(subset["spearman"].abs().sort_values(ascending=False).index).head(top_k)
        labels = [
            f"{driver} vs {proxy}"
            for driver, proxy in zip(subset["driver_column"], subset["proxy_column"])
        ]
        vals = subset["spearman"].to_numpy(dtype=float)
        colors = [family_colors.get(f, "#969696") for f in subset["proxy_family"]]

        fig_h = max(5.0, 0.30 * len(subset) + 1.5)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        y = np.arange(len(subset))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_xlabel("Spearman correlation")
        ax.set_title(f"Top precipitation/proxy correlations ({transform}, pooled targets, lag 0)")
        ax.invert_yaxis()
        fig.tight_layout()
        path = os.path.join(plot_dir, f"top_correlations_{transform}_lag0.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved: {path}")


def run(args: argparse.Namespace) -> tuple[str, str]:
    df = load_timeseries(args.input_dir)
    driver_columns = args.driver_columns or infer_driver_columns(df)
    if not driver_columns:
        raise RuntimeError("No precipitation driver columns found. Pass --driver-columns explicitly.")

    supplemental_precip = None
    supplemental_path = getattr(args, "supplemental_precip_file", None)
    if supplemental_path:
        supplemental_precip = pd.read_csv(
            supplemental_path,
            parse_dates=["init_time"],
        ).set_index("init_time").sort_index()
        required = {"target", *driver_columns}
        missing = sorted(required - set(supplemental_precip.columns))
        if missing:
            raise ValueError(
                f"Supplemental precipitation CSV is missing columns: {missing}"
            )

    out_dir = ensure_dir(os.path.join(args.output_dir, "correlations"))
    corr, seasonal = compute_correlations(
        df,
        driver_columns=driver_columns,
        transforms=args.transforms,
        lag_hours=args.lag_hours,
        min_n=args.min_n,
        supplemental_precip=supplemental_precip,
    )

    corr_path = os.path.join(out_dir, "correlations.csv")
    seasonal_path = os.path.join(out_dir, "seasonal_correlations.csv")
    corr.to_csv(corr_path, index=False)
    seasonal.to_csv(seasonal_path, index=False)
    print(f"Saved: {corr_path}")
    print(f"Saved: {seasonal_path}")

    write_summary(corr, out_dir, args.top_k)
    if not args.no_plots:
        maybe_plot(corr, out_dir, args.top_k)
    return corr_path, seasonal_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
