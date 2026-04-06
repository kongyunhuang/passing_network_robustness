"""
statistical_analysis.py - Shared publication statistics for the single-match pipeline
===================================================================================

This module provides the statistical analyses used by the paper after the pipeline
shift from season-aggregated team graphs to single-match graphs with season-level
summary by arithmetic averaging.

Core assumptions
----------------
1. `batch_summary.csv` stores team-season-venue summary rows obtained by averaging
   match-level indicators.
2. `match_summary.csv` stores the underlying match-level networks.
3. Structural analyses in Section 3.2 are still reported at the team level
   (mean over seasons and venues) unless otherwise specified.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg

    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from data_loader import (
    apply_substitution_map,
    build_substitution_map,
    extract_passes,
    get_match_files,
    safe_load_match_events,
)
from network_builder import ZonalNetworkBuilder
from robustness import (
    edge_removal_simulation,
    calculate_robustness_integral,
    find_critical_threshold,
)


STRUCTURAL_METRICS = [
    "AvgDegree",
    "Clustering",
    "Assortativity",
    "Reciprocity",
    "FlowHierarchy",
    "OutDegreeCV",
    "Mu11_Mu",
    "CyclicTriadFraction",
]

LEAGUE_NAME_TO_ID = {
    "Premier League": 2,
    "La Liga": 11,
}

SEASON_NAME_TO_ID = {
    "2022/23": 235,
    "2023/24": 281,
    "2024/25": 317,
}

# Set this to your local StatsBomb event data directory
DEFAULT_DATA_DIR = Path("data/raw")

_MATCH_FILE_INDEX: Dict[tuple[str, str], Dict[int, Path]] = {}
_BUILDER = ZonalNetworkBuilder()


def correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z confidence interval for Pearson r."""
    if n <= 3 or np.isnan(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def _safe_pearsonr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Pearson correlation with small-sample / constant-vector protection."""
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return (np.nan, np.nan)
    return stats.pearsonr(x, y)


def _match_level_scale_frame(df: pd.DataFrame, scale: str = "L2") -> pd.DataFrame:
    """Filter to single-match level at specified scale with SCGC and fc>0 filters.

    Returns the match-level DataFrame (N≈4343 for L2) used in the paper's
    §3.2 analysis.  Previous versions aggregated to team level (N=48) which
    inflated correlation coefficients — this was fixed on 2026-04-05.
    """
    # Determine the correct column name for attack threshold
    attack_col = None
    for col in ["fc_Attack_Strong", "pc_Attack_Strong"]:
        if col in df.columns:
            attack_col = col
            break
    if attack_col is None:
        raise KeyError("Neither fc_Attack_Strong nor pc_Attack_Strong found")

    # Determine the correct column name for error threshold (used for filtering)
    error_col = None
    for col in ["fc_Error_Strong", "pc_Error_Strong"]:
        if col in df.columns:
            error_col = col
            break

    required = STRUCTURAL_METRICS + [attack_col, "Scale"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_scale = df[df["Scale"] == scale].copy()

    # Apply SCGC≥0.3 and fc>0 filters (consistent with paper §3.1)
    if "SCGC_Ratio" in df_scale.columns:
        df_scale = df_scale[df_scale["SCGC_Ratio"] >= 0.3]
    if error_col:
        df_scale = df_scale[df_scale[error_col] > 0]
    if attack_col:
        df_scale = df_scale[df_scale[attack_col] > 0]

    df_scale = df_scale.dropna(subset=STRUCTURAL_METRICS + [attack_col])
    return df_scale


def _manual_partial_corr(df: pd.DataFrame, x: str, y: str, covars: Iterable[str]) -> tuple[float, float]:
    """Compute partial correlation via residualization when pingouin is absent."""
    from sklearn.linear_model import LinearRegression

    covars = list(covars)
    X_cov = df[covars].values
    lr = LinearRegression()

    lr.fit(X_cov, df[x].values)
    resid_x = df[x].values - lr.predict(X_cov)

    lr.fit(X_cov, df[y].values)
    resid_y = df[y].values - lr.predict(X_cov)

    return stats.pearsonr(resid_x, resid_y)


def full_correlation_table(df: pd.DataFrame, scale: str = "L2") -> pd.DataFrame:
    """Correlation table with effect sizes, confidence intervals, and Bonferroni correction."""
    df_agg = _match_level_scale_frame(df, scale=scale)
    rows = []
    for metric in STRUCTURAL_METRICS:
        r, p = _safe_pearsonr(df_agg[metric], df_agg["pc_Attack_Strong"])
        ci_low, ci_high = correlation_ci(r, len(df_agg))
        rows.append(
            {
                "metric": metric,
                "r": r,
                "r_squared": r**2,
                "CI_lower": ci_low,
                "CI_upper": ci_high,
                "p_value": p,
                "n": len(df_agg),
            }
        )
    result = pd.DataFrame(rows)
    n_tests = len(result)
    result["p_bonferroni"] = (result["p_value"] * n_tests).clip(upper=1.0)
    result["sig_uncorrected"] = result["p_value"] < 0.05
    result["sig_bonferroni"] = result["p_bonferroni"] < 0.05
    return result


def partial_correlation_analysis(df: pd.DataFrame, scale: str = "L2") -> pd.DataFrame:
    """Partial correlations for structural metrics vs attack robustness."""
    df_agg = _match_level_scale_frame(df, scale=scale)
    rows = []
    for metric in STRUCTURAL_METRICS:
        r_zero, p_zero = _safe_pearsonr(df_agg[metric], df_agg["pc_Attack_Strong"])
        covars = [name for name in STRUCTURAL_METRICS if name != metric]
        if len(df_agg) < len(covars) + 3:
            rows.append(
                {
                    "metric": metric,
                    "zero_order_r": r_zero,
                    "zero_order_p": p_zero,
                    "partial_r": np.nan,
                    "partial_p": np.nan,
                    "r_change": np.nan,
                }
            )
            continue
        if PINGOUIN_AVAILABLE:
            pcorr = pg.partial_corr(
                data=df_agg,
                x=metric,
                y="pc_Attack_Strong",
                covar=covars,
                method="pearson",
            )
            r_partial = pcorr["r"].values[0]
            p_partial = pcorr["p-val"].values[0]
        else:
            r_partial, p_partial = _manual_partial_corr(
                df_agg, metric, "pc_Attack_Strong", covars
            )

        rows.append(
            {
                "metric": metric,
                "zero_order_r": r_zero,
                "zero_order_p": p_zero,
                "partial_r": r_partial,
                "partial_p": p_partial,
                "r_change": abs(r_partial) - abs(r_zero),
            }
        )
    return pd.DataFrame(rows)


def multiple_regression_analysis(df: pd.DataFrame, scale: str = "L2") -> dict:
    """OLS regression of attack robustness on directed structural metrics."""
    if not STATSMODELS_AVAILABLE:
        return {}

    df_agg = _match_level_scale_frame(df, scale=scale)
    if len(df_agg) <= len(STRUCTURAL_METRICS) + 1:
        return {}
    X = df_agg[STRUCTURAL_METRICS]
    y = df_agg["pc_Attack_Strong"]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    x_sd = X.std()
    x_sd = x_sd.replace(0, np.nan)  # 零方差列 → NaN，避免除零
    X_std = (X - X.mean()) / x_sd
    y_sd = y.std()
    y_std = (y - y.mean()) / y_sd if y_sd > 0 else y - y.mean()
    X_std_const = sm.add_constant(X_std)
    model_std = sm.OLS(y_std, X_std_const).fit()

    rows = []
    for idx, predictor in enumerate(STRUCTURAL_METRICS, start=1):
        rows.append(
            {
                "predictor": predictor,
                "beta_raw": model.params[predictor],
                "beta_std": model_std.params[predictor],
                "se": model.bse[predictor],
                "t_stat": model.tvalues[predictor],
                "p_value": model.pvalues[predictor],
                "VIF": variance_inflation_factor(X_const.values, idx),
            }
        )

    return {
        "summary_table": pd.DataFrame(rows),
        "adj_r2": model.rsquared_adj,
        "r2": model.rsquared,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "n": len(df_agg),
        "aic": model.aic,
        "bic": model.bic,
    }


def _default_match_csv(summary_csv_path: Path) -> Path:
    return summary_csv_path.with_name("match_summary.csv")


def _get_match_file(league_name: str, season_name: str, match_id: int, data_dir: Path) -> Optional[Path]:
    key = (league_name, season_name)
    if key not in _MATCH_FILE_INDEX:
        league_id = LEAGUE_NAME_TO_ID.get(league_name)
        season_id = SEASON_NAME_TO_ID.get(season_name)
        if league_id is None or season_id is None:
            return None
        try:
            files = get_match_files(str(data_dir), league_id, season_id)
        except FileNotFoundError:
            return None
        _MATCH_FILE_INDEX[key] = {int(path.name.split("_")[0]): path for path in files}
    return _MATCH_FILE_INDEX[key].get(int(match_id))


def _rebuild_match_graph(row: pd.Series, data_dir: Path) -> Optional[nx.MultiDiGraph]:
    match_path = _get_match_file(row["League"], row["Season"], int(row["MatchID"]), data_dir)
    if match_path is None:
        return None

    events = safe_load_match_events(str(match_path))
    if events is None:
        return None

    team_id = int(row["TeamID"])
    passes = extract_passes(events, match_id=int(row["MatchID"]), team_id=team_id)
    if not passes:
        return None

    sub_map = build_substitution_map(events, team_id=team_id)
    apply_substitution_map(passes, sub_map)
    return _BUILDER.build_network(passes, scale=row["Scale"])


def _select_stratified_rows(df: pd.DataFrame, n_sample: int, seed: int) -> pd.DataFrame:
    """Select rows spanning the observed robustness range."""
    df_sorted = df.sort_values("pc_Attack_Strong").reset_index(drop=True)
    if len(df_sorted) <= n_sample:
        return df_sorted

    thirds = np.array_split(df_sorted.index.to_numpy(), 3)
    rng = random.Random(seed)
    selected_indices: List[int] = []
    remaining = n_sample
    for bucket in thirds:
        if remaining <= 0:
            break
        take = min(max(n_sample // 3, 1), len(bucket), remaining)
        selected_indices.extend(rng.sample(list(bucket), take))
        remaining -= take

    if remaining > 0:
        leftovers = [idx for idx in df_sorted.index if idx not in selected_indices]
        selected_indices.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    return df_sorted.loc[sorted(selected_indices)].reset_index(drop=True)


def configuration_model_null_test(
    df: pd.DataFrame,
    match_csv_path: Optional[str] = None,
    n_sample: int = 30,
    n_null: int = 20,
    scale: str = "L2",
    seed: int = 42,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Compare real single-match attack robustness against directed configuration nulls.

    The new pipeline samples match-level networks at a fixed scale, rebuilds the
    exact team-match graph from raw StatsBomb data, and compares the observed
    attack threshold/integral to null graphs with preserved in/out degree
    sequences.
    """
    if match_csv_path is None:
        if not hasattr(df, "attrs") or "source_path" not in df.attrs:
            raise ValueError("match_csv_path is required when df.attrs['source_path'] is unavailable.")
        match_csv = _default_match_csv(Path(df.attrs["source_path"]))
    else:
        match_csv = Path(match_csv_path)

    if not match_csv.exists():
        raise FileNotFoundError(f"Match summary not found: {match_csv}")

    match_df = pd.read_csv(match_csv)
    match_df = match_df[match_df["Scale"] == scale].dropna(subset=["pc_Attack_Strong", "R_Attack_Strong"])
    sampled = _select_stratified_rows(match_df, n_sample=n_sample, seed=seed)

    rng = np.random.default_rng(seed)
    rows = []

    for _, row in sampled.iterrows():
        graph = _rebuild_match_graph(row, data_dir=data_dir)
        if graph is None:
            continue

        G_dir = nx.DiGraph(graph)
        if G_dir.number_of_nodes() < 3 or G_dir.number_of_edges() < 3:
            continue

        in_seq = [deg for _, deg in G_dir.in_degree()]
        out_seq = [deg for _, deg in G_dir.out_degree()]
        if sum(in_seq) == 0 or sum(out_seq) == 0:
            continue

        null_pcs = []
        null_rs = []
        null_failures = 0
        for _ in range(n_null):
            try:
                G_null = nx.directed_configuration_model(
                    in_seq,
                    out_seq,
                    seed=int(rng.integers(0, 2**31)),
                )
                G_null = nx.DiGraph(G_null)
                G_null.remove_edges_from(nx.selfloop_edges(G_null))
                sim_attack = edge_removal_simulation(G_null, method="attack", n_steps=50)
                pc_null, _ = find_critical_threshold(sim_attack, "strong")
                r_null = calculate_robustness_integral(sim_attack, "strong")
                null_pcs.append(pc_null)
                null_rs.append(r_null)
            except Exception:
                null_failures += 1
                continue

        if null_failures > 0:
            print(f"  ⚠️ Null model: {null_failures}/{n_null} failed for "
                  f"Match={row['MatchID']} Team={row['Team']}")

        if not null_pcs:
            continue

        real_pc = float(row["pc_Attack_Strong"])
        real_r = float(row["R_Attack_Strong"])
        pc_mean = float(np.mean(null_pcs))
        pc_std = float(np.std(null_pcs, ddof=1)) if len(null_pcs) > 1 else 0.0
        r_mean = float(np.mean(null_rs))
        r_std = float(np.std(null_rs, ddof=1)) if len(null_rs) > 1 else 0.0
        rows.append(
            {
                "League": row["League"],
                "Season": row["Season"],
                "Team": row["Team"],
                "TeamID": row["TeamID"],
                "Venue": row["Venue"],
                "MatchID": row["MatchID"],
                "Scale": row["Scale"],
                "real_pc_Attack": real_pc,
                "null_pc_mean": pc_mean,
                "null_pc_std": pc_std,
                "pc_z_score": (real_pc - pc_mean) / max(pc_std, 1e-6),
                "real_R_Attack": real_r,
                "null_R_mean": r_mean,
                "null_R_std": r_std,
                "R_z_score": (real_r - r_mean) / max(r_std, 1e-6),
                "null_success_count": len(null_pcs),
                "null_total_attempts": n_null,
            }
        )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        t_pc, p_pc = stats.ttest_rel(result_df["real_pc_Attack"], result_df["null_pc_mean"])
        t_r, p_r = stats.ttest_rel(result_df["real_R_Attack"], result_df["null_R_mean"])
        result_df.attrs["paired_t_pc"] = (t_pc, p_pc)
        result_df.attrs["paired_t_r"] = (t_r, p_r)
    return result_df


def run_all_analyses(
    csv_path: str,
    scale: str = "L2",
    run_null_test: bool = False,
    match_csv_path: Optional[str] = None,
    null_n_sample: int = 30,
    null_n_null: int = 20,
) -> dict:
    """Run publication statistics and save output CSVs next to the input summary."""
    df = pd.read_csv(csv_path)
    df.attrs["source_path"] = csv_path

    corr_table = full_correlation_table(df, scale=scale)
    partials = partial_correlation_analysis(df, scale=scale)
    regression = multiple_regression_analysis(df, scale=scale)

    output_dir = Path(csv_path).resolve().parent
    corr_table.to_csv(output_dir / "stat_correlation_table.csv", index=False)
    partials.to_csv(output_dir / "stat_partial_correlations.csv", index=False)
    if regression:
        regression["summary_table"].to_csv(output_dir / "stat_regression_table.csv", index=False)

    null_df = pd.DataFrame()
    if run_null_test:
        null_df = configuration_model_null_test(
            df,
            match_csv_path=match_csv_path,
            n_sample=null_n_sample,
            n_null=null_n_null,
            scale=scale,
        )
        if not null_df.empty:
            null_df.to_csv(output_dir / "null_model_results.csv", index=False)

    return {
        "correlation_table": corr_table,
        "partial_correlations": partials,
        "regression": regression,
        "null_model": null_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper statistical analyses.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "output" / "results" / "batch_summary.csv"),
        help="Path to batch_summary.csv",
    )
    parser.add_argument("--scale", default="L2", help="Scale to analyze (default: L2)")
    parser.add_argument("--run-null-test", action="store_true", help="Also run configuration-model null test.")
    parser.add_argument(
        "--match-csv",
        default=None,
        help="Optional path to match_summary.csv; defaults to sibling of batch_summary.csv.",
    )
    parser.add_argument("--null-sample", type=int, default=30, help="Number of match networks sampled for null test.")
    parser.add_argument("--null-realizations", type=int, default=20, help="Null realizations per sampled match.")
    args = parser.parse_args()

    results = run_all_analyses(
        csv_path=args.csv_path,
        scale=args.scale,
        run_null_test=args.run_null_test,
        match_csv_path=args.match_csv,
        null_n_sample=args.null_sample,
        null_n_null=args.null_realizations,
    )

    print("=" * 70)
    print("  Statistical Analysis for Publication")
    print("=" * 70)
    print(results["correlation_table"].to_string(index=False, float_format="%.4f"))
    print("\n" + "=" * 70)
    print("  Partial Correlations")
    print("=" * 70)
    print(results["partial_correlations"].to_string(index=False, float_format="%.4f"))
    if results["regression"]:
        print("\n" + "=" * 70)
        print("  Multiple Regression")
        print("=" * 70)
        print(results["regression"]["summary_table"].to_string(index=False, float_format="%.4f"))
    if args.run_null_test:
        print("\n" + "=" * 70)
        print("  Configuration Null Test")
        print("=" * 70)
        if results["null_model"].empty:
            print("  No valid null-model rows produced.")
        else:
            print(results["null_model"].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
