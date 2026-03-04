from __future__ import annotations

import argparse
from pathlib import Path

from report_utils import run_full_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Online News Popularity final project analysis."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("OnlineNewsPopularity/OnlineNewsPopularity.csv"),
        help="Path to the Online News Popularity CSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Directory where figures and tables will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_full_analysis(args.data, args.outdir)
    print(f"Analysis complete for {artifacts.sample_size} articles.")
    print(f"Missing values: {artifacts.missing_total}")
    print(f"Viral threshold (90th percentile): {artifacts.threshold} shares")
    print(
        "KS test on standardized log-shares: "
        f"statistic={artifacts.ks_statistic:.4f}, p-value={artifacts.ks_pvalue:.4g}"
    )
    print(
        "Skewness improvement: "
        f"raw={artifacts.raw_skew:.3f}, log1p={artifacts.log_skew:.3f}"
    )
    print(
        "Model fit summaries: "
        f"OLS R^2={artifacts.ols_results.rsquared:.4f}, "
        f"McFadden R^2={artifacts.mcfadden_r2:.4f}, AUC={artifacts.auc:.4f}"
    )
    print(f"Outputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
