from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PREDICTOR_COLS = [
    "n_tokens_title",
    "n_tokens_content",
    "num_imgs",
    "num_videos",
    "global_sentiment_polarity",
    "global_subjectivity",
    "rate_positive_words",
    "rate_negative_words",
]

CHANNEL_COLS = [
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]

REFERENCE_CHANNEL = "world"
FORMULA_BASE = (
    "z_n_tokens_title + z_n_tokens_content + z_num_imgs + z_num_videos + "
    "is_weekend + z_global_sentiment_polarity + z_global_subjectivity + "
    "z_rate_positive_words + z_rate_negative_words + "
    'C(channel, Treatment(reference="world"))'
)


@dataclass
class AnalysisArtifacts:
    table1: pd.DataFrame
    table2: pd.DataFrame
    table3: pd.DataFrame
    table4: pd.DataFrame
    table5: pd.DataFrame
    ols_results: object
    ols_robust: object
    ols_interaction: object
    logit_results: object
    threshold: int
    sample_size: int
    missing_total: int
    ks_statistic: float
    ks_pvalue: float
    raw_skew: float
    log_skew: float
    mcfadden_r2: float
    auc: float


def configure_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
        }
    )


def ensure_output_dirs(outdir: Path) -> tuple[Path, Path]:
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(data_path, skipinitialspace=True)
    threshold = int(df["shares"].quantile(0.9))
    df["log_shares"] = np.log1p(df["shares"])
    df["viral_top10"] = (df["shares"] >= threshold).astype(int)
    df["channel"] = "other"

    for column in CHANNEL_COLS:
        label = column.replace("data_channel_is_", "")
        df.loc[df[column] == 1, "channel"] = label

    for column in PREDICTOR_COLS:
        mean = df[column].mean()
        std = df[column].std()
        df[f"z_{column}"] = (df[column] - mean) / std

    return df, threshold


def median_iqr(series: pd.Series) -> tuple[float, float, float]:
    return (
        float(series.median()),
        float(series.quantile(0.25)),
        float(series.quantile(0.75)),
    )


def rank_biserial_from_u(u_stat: float, n1: int, n0: int) -> float:
    return (2 * u_stat) / (n1 * n0) - 1


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    if n == 0 or min_dim <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min_dim)))


def build_table1(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            {
                "section": "overview",
                "metric": "analytic_sample_size",
                "group": "all",
                "value": len(df),
            },
            {
                "section": "overview",
                "metric": "missing_values_total",
                "group": "all",
                "value": int(df.isna().sum().sum()),
            },
            {
                "section": "overview",
                "metric": "viral_threshold_shares",
                "group": "all",
                "value": threshold,
            },
            {
                "section": "overview",
                "metric": "viral_rate",
                "group": "all",
                "value": round(float(df["viral_top10"].mean()), 4),
            },
            {
                "section": "overview",
                "metric": "uci_metadata_rows",
                "group": "all",
                "value": 39797,
            },
            {
                "section": "overview",
                "metric": "local_csv_rows",
                "group": "all",
                "value": len(df),
            },
        ]
    )

    for group_name, group_df in [("all", df), ("non_viral", df[df["viral_top10"] == 0]), ("viral", df[df["viral_top10"] == 1])]:
        for column in ["shares", "log_shares", "n_tokens_content", "num_imgs", "num_videos", "global_subjectivity"]:
            median, q1, q3 = median_iqr(group_df[column])
            rows.extend(
                [
                    {
                        "section": "distribution",
                        "metric": f"{column}_median",
                        "group": group_name,
                        "value": round(median, 4),
                    },
                    {
                        "section": "distribution",
                        "metric": f"{column}_q1",
                        "group": group_name,
                        "value": round(q1, 4),
                    },
                    {
                        "section": "distribution",
                        "metric": f"{column}_q3",
                        "group": group_name,
                        "value": round(q3, 4),
                    },
                ]
            )

    channel_rates = (
        df.groupby("channel", as_index=False)["viral_top10"]
        .mean()
        .rename(columns={"viral_top10": "value"})
    )
    for record in channel_rates.to_dict("records"):
        rows.append(
            {
                "section": "channel_rates",
                "metric": "viral_rate",
                "group": record["channel"],
                "value": round(float(record["value"]), 4),
            }
        )

    return pd.DataFrame(rows)


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    pvalues: list[float] = []

    viral = df[df["viral_top10"] == 1]
    non_viral = df[df["viral_top10"] == 0]

    for column in [
        "n_tokens_title",
        "n_tokens_content",
        "num_imgs",
        "num_videos",
        "global_sentiment_polarity",
        "global_subjectivity",
        "rate_positive_words",
        "rate_negative_words",
    ]:
        u_stat, pvalue = stats.mannwhitneyu(
            viral[column],
            non_viral[column],
            alternative="two-sided",
        )
        viral_median, viral_q1, viral_q3 = median_iqr(viral[column])
        non_median, non_q1, non_q3 = median_iqr(non_viral[column])
        effect_size = rank_biserial_from_u(u_stat, len(viral), len(non_viral))
        pvalues.append(float(pvalue))
        results.append(
            {
                "variable": column,
                "test_type": "mann_whitney_u",
                "viral_summary": f"{viral_median:.3f} [{viral_q1:.3f}, {viral_q3:.3f}]",
                "non_viral_summary": f"{non_median:.3f} [{non_q1:.3f}, {non_q3:.3f}]",
                "test_statistic": float(u_stat),
                "effect_size": float(effect_size),
                "effect_size_label": "rank_biserial_r",
                "p_value": float(pvalue),
            }
        )

    for column in ["is_weekend", "channel"]:
        table = pd.crosstab(df["viral_top10"], df[column])
        chi2, pvalue, _, _ = stats.chi2_contingency(table)
        pvalues.append(float(pvalue))

        if column == "is_weekend":
            summaries = {
                "viral": round(float(viral[column].mean()), 4),
                "non_viral": round(float(non_viral[column].mean()), 4),
            }
        else:
            viral_props = viral[column].value_counts(normalize=True).sort_index()
            non_props = non_viral[column].value_counts(normalize=True).sort_index()
            summaries = {
                "viral": "; ".join(
                    f"{idx}:{value:.3f}" for idx, value in viral_props.items()
                ),
                "non_viral": "; ".join(
                    f"{idx}:{value:.3f}" for idx, value in non_props.items()
                ),
            }

        results.append(
            {
                "variable": column,
                "test_type": "chi_square",
                "viral_summary": summaries["viral"],
                "non_viral_summary": summaries["non_viral"],
                "test_statistic": float(chi2),
                "effect_size": cramers_v(table),
                "effect_size_label": "cramers_v",
                "p_value": float(pvalue),
            }
        )

    adjusted = multipletests(pvalues, method="fdr_bh")[1]
    for idx, qvalue in enumerate(adjusted):
        results[idx]["fdr_bh_q_value"] = float(qvalue)

    return pd.DataFrame(results).sort_values(["test_type", "p_value"]).reset_index(drop=True)


def fit_models(df: pd.DataFrame) -> tuple[object, object, object, object]:
    ols_formula = f"log_shares ~ {FORMULA_BASE}"
    interaction_formula = (
        f"{ols_formula} + "
        'C(channel, Treatment(reference="world")):z_global_sentiment_polarity + '
        'C(channel, Treatment(reference="world")):z_global_subjectivity'
    )
    logit_formula = f"viral_top10 ~ {FORMULA_BASE}"

    ols_results = smf.ols(ols_formula, data=df).fit()
    ols_robust = ols_results.get_robustcov_results(cov_type="HC3")
    ols_interaction = smf.ols(interaction_formula, data=df).fit()
    logit_results = smf.logit(logit_formula, data=df).fit(disp=False)
    return ols_results, ols_robust, ols_interaction, logit_results


def build_table3(ols_robust: object, logit_results: object) -> pd.DataFrame:
    ols_terms = pd.DataFrame(
        {
            "model": "ols_log_shares_hc3",
            "term": ols_robust.model.exog_names,
            "estimate": ols_robust.params,
            "std_error": ols_robust.bse,
            "ci_low": ols_robust.conf_int()[:, 0],
            "ci_high": ols_robust.conf_int()[:, 1],
            "p_value": ols_robust.pvalues,
            "estimate_type": "coefficient",
        }
    )

    conf = logit_results.conf_int()
    logit_terms = pd.DataFrame(
        {
            "model": "logit_viral_top10",
            "term": logit_results.params.index,
            "estimate": np.exp(logit_results.params.values),
            "std_error": logit_results.bse.values,
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": logit_results.pvalues.values,
            "estimate_type": "odds_ratio",
        }
    )

    return pd.concat([ols_terms, logit_terms], ignore_index=True)


def calculate_vif(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    design = df[list(columns)].copy()
    design["intercept"] = 1.0
    rows = []
    for idx, column in enumerate(design.columns[:-1]):
        rows.append(
            {
                "metric_group": "vif",
                "metric": column,
                "value": float(variance_inflation_factor(design.values, idx)),
            }
        )
    return pd.DataFrame(rows)


def build_table4(
    df: pd.DataFrame,
    ols_results: object,
    ols_robust: object,
    logit_results: object,
) -> tuple[pd.DataFrame, float, float]:
    diagnostics: list[dict[str, object]] = []

    bp_test = het_breuschpagan(ols_results.resid, ols_results.model.exog)
    diagnostics.extend(
        [
            {"metric_group": "ols_diagnostic", "metric": "breusch_pagan_lm", "value": float(bp_test[0])},
            {"metric_group": "ols_diagnostic", "metric": "breusch_pagan_pvalue", "value": float(bp_test[1])},
            {"metric_group": "ols_diagnostic", "metric": "ols_r_squared", "value": float(ols_results.rsquared)},
            {
                "metric_group": "logit_diagnostic",
                "metric": "mcfadden_pseudo_r2",
                "value": float(logit_results.prsquared),
            },
        ]
    )

    auc = roc_auc_score(df["viral_top10"], logit_results.predict(df))
    diagnostics.append(
        {"metric_group": "logit_diagnostic", "metric": "roc_auc", "value": float(auc)}
    )

    vif_table = calculate_vif(
        df,
        [
            "z_n_tokens_title",
            "z_n_tokens_content",
            "z_num_imgs",
            "z_num_videos",
            "is_weekend",
            "z_global_sentiment_polarity",
            "z_global_subjectivity",
            "z_rate_positive_words",
            "z_rate_negative_words",
        ],
    )
    diagnostics.extend(vif_table.to_dict("records"))

    cutoff = df["shares"].quantile(0.99)
    trimmed_df = df[df["shares"] < cutoff].copy()
    trimmed_ols = smf.ols(f"log_shares ~ {FORMULA_BASE}", data=trimmed_df).fit()
    trimmed_robust = trimmed_ols.get_robustcov_results(cov_type="HC3")

    key_terms = [
        "z_n_tokens_title",
        "z_n_tokens_content",
        "z_num_imgs",
        "z_num_videos",
        "is_weekend",
    ]
    term_index = {term: idx for idx, term in enumerate(ols_robust.model.exog_names)}
    trimmed_index = {term: idx for idx, term in enumerate(trimmed_robust.model.exog_names)}

    for term in key_terms:
        main_coef = float(ols_robust.params[term_index[term]])
        trimmed_coef = float(trimmed_robust.params[trimmed_index[term]])
        diagnostics.append(
            {
                "metric_group": "robustness",
                "metric": term,
                "value": trimmed_coef,
                "baseline_value": main_coef,
                "same_direction": np.sign(main_coef) == np.sign(trimmed_coef),
            }
        )

    return pd.DataFrame(diagnostics), float(logit_results.prsquared), float(auc)


def build_table5(ols_results: object, ols_interaction: object) -> pd.DataFrame:
    nested = anova_lm(ols_results, ols_interaction)
    return pd.DataFrame(
        [
            {
                "comparison": "base_vs_interaction_ols",
                "df_diff": float(nested.loc[1, "df_diff"]),
                "ss_diff": float(nested.loc[1, "ss_diff"]),
                "f_statistic": float(nested.loc[1, "F"]),
                "p_value": float(nested.loc[1, "Pr(>F)"]),
            }
        ]
    )


def generate_fig1(df: pd.DataFrame, figure_path: Path) -> tuple[float, float, float, float]:
    standardized_log = (df["log_shares"] - df["log_shares"].mean()) / df["log_shares"].std()
    ks_statistic, ks_pvalue = stats.kstest(standardized_log, "norm")
    raw_skew = float(stats.skew(df["shares"]))
    log_skew = float(stats.skew(df["log_shares"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["shares"], bins=60, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Raw Shares")
    axes[0].set_xlabel("Shares")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, df["shares"].quantile(0.99))

    sns.histplot(df["log_shares"], bins=50, ax=axes[1], color="#F58518")
    axes[1].set_title("Log-Transformed Shares")
    axes[1].set_xlabel("log(1 + shares)")
    axes[1].set_ylabel("Count")

    qqplot(df["log_shares"], line="45", ax=axes[2], alpha=0.5)
    axes[2].set_title("Q-Q Plot of log(1 + shares)")
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return ks_statistic, ks_pvalue, raw_skew, log_skew


def generate_fig2(df: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    channel_rates = (
        df.groupby("channel")["viral_top10"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
        .reset_index()
    )
    sns.barplot(
        data=channel_rates,
        x="channel",
        y="viral_top10",
        hue="channel",
        dodge=False,
        legend=False,
        ax=axes[0],
        palette="crest",
    )
    axes[0].set_title("Viral Rate by Channel")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Percent viral")
    axes[0].tick_params(axis="x", rotation=25)

    weekend_rates = (
        df.assign(day_type=np.where(df["is_weekend"] == 1, "Weekend", "Weekday"))
        .groupby("day_type")["viral_top10"]
        .mean()
        .mul(100)
        .reset_index()
    )
    sns.barplot(
        data=weekend_rates,
        x="day_type",
        y="viral_top10",
        hue="day_type",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="flare",
    )
    axes[1].set_title("Viral Rate by Publication Timing")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Percent viral")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def generate_fig3(df: pd.DataFrame, ols_interaction: object, figure_path: Path) -> None:
    subjectivity_range = np.linspace(df["z_global_subjectivity"].quantile(0.05), df["z_global_subjectivity"].quantile(0.95), 80)
    channels = ["world", "lifestyle", "socmed", "tech", "entertainment", "bus", "other"]

    prediction_rows = []
    for channel in channels:
        temp = pd.DataFrame(
            {
                "z_n_tokens_title": 0.0,
                "z_n_tokens_content": 0.0,
                "z_num_imgs": 0.0,
                "z_num_videos": 0.0,
                "is_weekend": 0.0,
                "z_global_sentiment_polarity": 0.0,
                "z_global_subjectivity": subjectivity_range,
                "z_rate_positive_words": 0.0,
                "z_rate_negative_words": 0.0,
                "channel": channel,
            }
        )
        predicted = ols_interaction.get_prediction(temp).summary_frame(alpha=0.05)
        temp["predicted_log_shares"] = predicted["mean"]
        temp["ci_low"] = predicted["mean_ci_lower"]
        temp["ci_high"] = predicted["mean_ci_upper"]
        prediction_rows.append(temp)

    predictions = pd.concat(prediction_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.lineplot(
        data=predictions,
        x="z_global_subjectivity",
        y="predicted_log_shares",
        hue="channel",
        ax=ax,
    )
    ax.set_title("Adjusted Relationship Between Subjectivity and Expected log(1 + shares)")
    ax.set_xlabel("Standardized global subjectivity")
    ax.set_ylabel("Predicted log(1 + shares)")
    ax.legend(title="Channel", ncol=2)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def generate_fig4(ols_results: object, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(
        x=ols_results.fittedvalues,
        y=ols_results.resid,
        ax=axes[0],
        s=14,
        alpha=0.35,
        edgecolor=None,
        color="#4C78A8",
    )
    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")

    qqplot(ols_results.resid, line="45", ax=axes[1], alpha=0.4)
    axes[1].set_title("Q-Q Plot of OLS Residuals")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Residual Quantiles")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def run_full_analysis(data_path: Path, outdir: Path) -> AnalysisArtifacts:
    configure_style()
    figures_dir, tables_dir = ensure_output_dirs(outdir)
    df, threshold = load_and_prepare_data(data_path)

    table1 = build_table1(df, threshold)
    table2 = build_table2(df)
    ols_results, ols_robust, ols_interaction, logit_results = fit_models(df)
    table3 = build_table3(ols_robust, logit_results)
    table4, mcfadden_r2, auc = build_table4(df, ols_results, ols_robust, logit_results)
    table5 = build_table5(ols_results, ols_interaction)

    ks_statistic, ks_pvalue, raw_skew, log_skew = generate_fig1(
        df, figures_dir / "fig1_share_distribution.png"
    )
    generate_fig2(df, figures_dir / "fig2_virality_rates.png")
    generate_fig3(df, ols_interaction, figures_dir / "fig3_interaction_subjectivity.png")
    generate_fig4(ols_results, figures_dir / "fig4_ols_diagnostics.png")

    table1.to_csv(tables_dir / "table1_data_summary.csv", index=False)
    table2.to_csv(tables_dir / "table2_univariate_tests.csv", index=False)
    table3.to_csv(tables_dir / "table3_model_results.csv", index=False)
    table4.to_csv(tables_dir / "table4_diagnostics.csv", index=False)
    table5.to_csv(tables_dir / "table5_interaction_test.csv", index=False)

    return AnalysisArtifacts(
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
        table5=table5,
        ols_results=ols_results,
        ols_robust=ols_robust,
        ols_interaction=ols_interaction,
        logit_results=logit_results,
        threshold=threshold,
        sample_size=len(df),
        missing_total=int(df.isna().sum().sum()),
        ks_statistic=float(ks_statistic),
        ks_pvalue=float(ks_pvalue),
        raw_skew=raw_skew,
        log_skew=log_skew,
        mcfadden_r2=mcfadden_r2,
        auc=auc,
    )
