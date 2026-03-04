# Final Project Reproducibility

This repository contains a complete, script-first submission package for the Math 189 final project, *Anatomy of a Viral Article*.

## Run the analysis

Install the dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/run_analysis.py --data OnlineNewsPopularity/OnlineNewsPopularity.csv --outdir outputs
```

The analysis script reads the raw UCI/Mashable CSV, creates the derived variables described in the report, fits the univariate tests and regression models, and regenerates all figures and tables.

## Generated outputs

Running the script recreates:

- `outputs/figures/fig1_share_distribution.png`
- `outputs/figures/fig2_virality_rates.png`
- `outputs/figures/fig3_interaction_subjectivity.png`
- `outputs/figures/fig4_ols_diagnostics.png`
- `outputs/tables/table1_data_summary.csv`
- `outputs/tables/table2_univariate_tests.csv`
- `outputs/tables/table3_model_results.csv`
- `outputs/tables/table4_diagnostics.csv`
- `outputs/tables/table5_interaction_test.csv`

The script also prints the key run-level checks needed for grading:

- analytic sample size: `39,644`
- missing values: `0`
- viral threshold: `6,200` shares

## Project layout

- `analysis/run_analysis.py`: command-line entrypoint
- `analysis/report_utils.py`: preprocessing, statistical analysis, and plotting utilities
- `report/final_report.md`: canonical written report
- `OnlineNewsPopularity/`: raw dataset files supplied for the project

## Notes

- The report is written in Markdown so it can be converted to PDF or Word if needed.
- The local CSV is treated as the authoritative analytic file. It contains `39,644` rows even though the UCI metadata documentation reports `39,797`.
- The analysis is observational and inference-focused; it does not benchmark machine-learning prediction models.
