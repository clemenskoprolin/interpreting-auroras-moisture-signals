"""
analyze_stratified.py — decomposition and moisture-stratified analysis of v6.

Exact score decomposition (holds per case and lead by construction):

    S(true) - S(qhat_sp) = [S(true)       - S(meanmatched)]   pattern effect
                         + [S(meanmatched) - S(qhat_sp)]      local-offset effect

Reported in ABSOLUTE units (g/kg) as the primary statistic: the dry strata have
small q850 box means, so relative percentages there have noisy denominators.

The regression separates the three explanations for a moisture dependence:

    delta ~ resid_mean + resid_std + tcwv_pct + tcwv_pct:resid_mean  (+ target FE)

  1. moist cases merely carry larger perturbations -> resid terms significant,
     tcwv_pct and the interaction not;
  2. Aurora is more sensitive per mm when moist -> interaction significant;
  3. sensitivity flat, residual larger in extremes -> resid terms significant,
     interaction not, and the stratum means track resid_mean.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "02_zwd_attribution_benchmark"))

MTN = ["andes", "himalayas", "rockies", "cascades_sierra", "caucasus",
       "ticino", "alps_east", "atlas", "valais", "pyrenees", "new_zealand_alps"]
STRAT_ORDER = ["low", "typical", "humid", "extreme"]


def ols(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Plain OLS with HC0 robust standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC0 sandwich
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    from scipy import stats as st
    p = 2 * (1 - st.norm.cdf(np.abs(t)))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=names)
    out.attrs["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    results_root = os.environ.get(
        "AURORA_XAI_RESULTS_DIR", os.path.join(os.path.dirname(_HERE), "results")
    )
    ap.add_argument("--results", nargs="+", default=[os.path.join(
        results_root, "zwd_counterfactual_interventions", "ablation_scores.csv")],
        help="One or more Stage-A ablation CSVs to combine.")
    ap.add_argument("--meta", nargs="+", default=[
        os.path.join(_HERE, "results", "cases_stratified96_meta.csv"),
        os.path.join(_HERE, "results", "cases_stratified_extension84_meta.csv"),
        os.path.join(_HERE, "results", "cases_stratified_flat_extension48_meta.csv"),
    ],
        help="One or more matching stratified-case metadata CSVs.")
    ap.add_argument("--out", default=os.path.join(
        _HERE, "results", "stratified_decomposition.csv"))
    ap.add_argument("--lead", type=int, default=None,
                    help="Restrict to one lead (default: pool 6/12/24).")
    args = ap.parse_args()

    d = pd.concat([pd.read_csv(path) for path in args.results], ignore_index=True)
    if args.lead:
        d = d[d.lead_h == args.lead]
    m = pd.concat([pd.read_csv(path) for path in args.meta], ignore_index=True)
    m["case_key"] = m.target + "|" + m.init_time
    duplicate_results = d.duplicated(
        ["target", "case_id", "init_time", "lead_h", "mode"])
    if duplicate_results.any():
        raise ValueError(
            f"{int(duplicate_results.sum())} duplicate Stage-A result rows "
            "across --results inputs")
    duplicate_meta = m.duplicated("case_key")
    if duplicate_meta.any():
        raise ValueError(
            f"{int(duplicate_meta.sum())} duplicate cases across --meta inputs")

    idx = ["target", "case_id", "init_time", "lead_h"]
    val = ["score"] + [c for c in d.columns if c.startswith("score_q850")]
    piv = d.pivot_table(index=idx, columns="mode", values="score").reset_index()
    have = [c for c in ["true", "qhat_sp", "qhat_sp_meanmatched",
                        "qhat_sp_regional", "qhat_tcwv"] if c in piv.columns]
    print(f"modes present: {have}")

    piv["total"] = piv["true"] - piv["qhat_sp"]
    if "qhat_sp_meanmatched" in piv:
        piv["pattern"] = piv["true"] - piv["qhat_sp_meanmatched"]
        piv["offset"] = piv["qhat_sp_meanmatched"] - piv["qhat_sp"]
        err = float(np.abs(piv.total - (piv.pattern + piv.offset)).max())
        print(f"decomposition identity max error: {err:.3e} (should be ~0)")
    if "qhat_tcwv" in piv:
        piv["total_tcwv"] = piv["true"] - piv["qhat_tcwv"]
    if "qhat_sp_regional" in piv:
        piv["total_regional"] = piv["true"] - piv["qhat_sp_regional"]

    piv["case_key"] = piv.target + "|" + piv.init_time
    df = piv.merge(m[["case_key", "stratum", "tcwv", "tcwv_pct", "precip",
                      "precip_pct", "resid_rms", "resid_mean", "resid_std"]],
                   on="case_key", how="left")
    df["terrain"] = np.where(df.target.isin(MTN), "mountain", "flat")
    df["stratum"] = pd.Categorical(df.stratum, STRAT_ORDER, ordered=True)

    cols = [c for c in ["total", "pattern", "offset", "total_tcwv", "total_regional"]
            if c in df.columns]
    print("\n=== decomposition by moisture stratum (absolute, g/kg) ===")
    print(df.groupby("stratum", observed=True)[cols + ["resid_mean", "resid_rms"]]
            .mean().round(4).to_string())

    print("\n=== fraction of the total effect carried by the local offset ===")
    g = df.groupby("stratum", observed=True)[["total", "offset"]].mean()
    print((g.offset / g.total).round(3).to_string())

    print("\n=== by terrain x stratum (total effect) ===")
    print(df.pivot_table(index="terrain", columns="stratum",
                         values="total", observed=True).round(4).to_string())

    print("\n=== moisture x precipitation cross-tab (total effect) ===")
    df["rain"] = np.select(
        [
            df.precip_pct.isna(),
            df.precip_pct >= 90,
            df.precip_pct >= 60,
        ],
        ["unavailable", "heavy", "moderate"],
        default="non-raining",
    )
    print(df.pivot_table(index="stratum", columns="rain", values="total",
                         observed=True, aggfunc="mean").round(4).to_string())
    print("\ncase counts:")
    print(df.pivot_table(index="stratum", columns="rain", values="total",
                         observed=True, aggfunc="size").to_string())

    if "score_q850_ag" in d.columns:
        print("\n=== above-ground vs unmasked q850 (mountain targets) ===")
        ag = d[d.target.isin(MTN)].pivot_table(
            index=["target", "case_id", "lead_h"], columns="mode",
            values=["score", "score_q850_ag"]).reset_index()
        for lbl, key in (("unmasked", "score"), ("above-ground", "score_q850_ag")):
            if (key, "true") in ag.columns and (key, "qhat_sp") in ag.columns:
                v = (ag[(key, "true")] - ag[(key, "qhat_sp")])
                print(f"  {lbl:13s} mean total effect = {v.mean():+.4f} g/kg  (n={v.notna().sum()})")

    print("\n=== regression: total ~ resid_mean + resid_std + tcwv_pct + interaction ===")
    reg = df.dropna(subset=["total", "resid_mean", "resid_std", "tcwv_pct"]).copy()
    z = lambda s: (s - s.mean()) / s.std()
    reg["rm"], reg["rs"], reg["tp"] = z(reg.resid_mean), z(reg.resid_std), z(reg.tcwv_pct)
    reg["ix"] = reg.rm * reg.tp
    names = ["const", "resid_mean", "resid_std", "tcwv_pct", "tcwv_pct:resid_mean"]
    X = np.column_stack([np.ones(len(reg)), reg.rm, reg.rs, reg.tp, reg.ix])
    res = ols(X, reg.total.values, names)
    print(res.round(4).to_string())
    print(f"  R2 = {res.attrs['r2']:.3f}   n = {len(reg)}")

    # With target fixed effects: does anything survive within-target variation?
    tg = pd.get_dummies(reg.target, drop_first=True).astype(float)
    Xf = np.column_stack([np.ones(len(reg)), reg.rm, reg.rs, reg.tp, reg.ix, tg.values])
    resf = ols(Xf, reg.total.values, names + list(tg.columns))
    print("\n  with target fixed effects:")
    print(resf.loc[names].round(4).to_string())
    print(f"  R2 = {resf.attrs['r2']:.3f}")

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
