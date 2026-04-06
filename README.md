# Scale-dependent Applicability of Directed Mean-field Percolation in Soccer Passing Networks

Code and data for reproducing the results in the paper.

## Project Structure

```
passing_network_robustness/
├── README.md
├── reproduce.ipynb                    # Reproducibility notebook
├── figures/                           # Paper figures
│   ├── fig1–fig8.png/pdf              # Main text Figures 1–8
│   └── figS1–figS2.png/pdf           # Supplementary Figures S1–S2
├── data/                              # Pre-computed data files
│   ├── match_summary.csv              # Primary dataset (18,240 rows: 4 scales × 4,560 networks)
│   ├── synthetic_baselines.csv        # Synthetic network baselines (ER/BA/WS/RGG)
│   ├── player_impact_*.csv            # Player structural impact (3 seasons)
│   ├── market_value_validation_all_seasons.csv  # Transfermarkt value matching
│   ├── transfermarkt_*.csv            # Transfermarkt player valuations
│   ├── position_node_records.csv      # Node-level positional degree data
│   ├── position_network_records.csv   # Network-level positional degree data
│   ├── batch_summary.csv              # Team–season–venue–scale arithmetic means (960 rows)
│   ├── stat_correlation_table.csv     # §3.2 Bivariate correlations (N=4,343)
│   ├── stat_partial_correlations.csv  # §3.2 Partial correlations (N=4,343)
│   └── stat_regression_table.csv      # §3.2 OLS multivariate regression (N=4,343)
└── code/
    ├── pipeline/                      # Core data-generation pipeline (JSON → match_summary.csv)
    │   ├── batch_runner.py            #   Orchestrator: iterates over all matches
    │   ├── data_loader.py             #   StatsBomb JSON parsing, pass extraction, substitution inheritance
    │   ├── network_builder.py         #   Multi-scale position-dependent directed network construction
    │   ├── robustness.py              #   Bond percolation Monte Carlo simulation + MR analytical threshold
    │   └── compute_directed_indicators.py  # Directed structural indicators
    ├── fig_multiscale.py              # → Fig 1
    ├── fig_error_vs_attack.py         # → Fig 2
    ├── fig_structural_factors.py      # → Fig 3
    ├── fig_partial_correlation.py     # → Fig 4
    ├── fig_mr_benchmark.py            # → Fig 5
    ├── fig_rai_scatter.py             # → Fig 6
    ├── fig_team_stability.py          # → Fig 7
    ├── fig_baena_case.py              # → Fig 8
    ├── fig_corr_heatmap.py            # → Fig S1
    ├── fig_position_analysis.py       # → Fig S2
    ├── statistical_analysis.py        # → Table 2 + stat_*.csv
    ├── cross_sectional_validation.py  # → Table 4
    ├── threshold_sensitivity.py       # → Table S4
    └── scgc_vs_wgcc.py               # → Table S5
```

## Data-Generation Pipeline

`code/pipeline/` contains the complete code for generating `match_summary.csv` from raw StatsBomb event data.

### Pipeline Flow

```
StatsBomb event data (2 leagues × 3 seasons × ~380 matches)
    ↓  data_loader.py: parse JSON, extract successful open-play passes, handle substitution inheritance
    ↓  network_builder.py: build position-dependent directed passing networks at 4 scales (L0–L3)
    ↓  compute_directed_indicators.py: compute reciprocity, flow hierarchy, triadic census, etc.
    ↓  robustness.py: bond percolation Monte Carlo (L2 = 20 trials, others = 5) + MR analytical threshold
    ↓  batch_runner.py: orchestrate all steps, export CSVs
    ↓
match_summary.csv (18,240 rows: 4 scales × 4,560 networks)
batch_summary.csv (960 rows: team–season–venue–scale aggregates)
```

### Running the Pipeline (Regenerating Data from Scratch)

```bash
# 1. Obtain StatsBomb event data (https://statsbomb.com/)
#    Structure: datos/{comp_id}_{season_id}_*/{match_id}_events.json

# 2. Set DATA_ROOT in code/pipeline/batch_runner.py to your local data path

# 3. Run
cd code/pipeline
python batch_runner.py          # Full run (~2–4 hours)
python batch_runner.py --test   # Test mode (EPL 22/23, first 50 matches)
```

Note: The full run is computationally intensive (4,560 matches × 4 scales × multiple MC trials). Pre-computed results are included in `data/match_summary.csv`, so all downstream analyses can be reproduced without re-running the pipeline.

## Quick Start

### Requirements

```bash
# Python 3.10+
# Key dependencies: pandas, numpy, scipy, matplotlib, seaborn, pingouin, statsmodels, scikit-learn, networkx
```

### One-Click Verification (Using Pre-Computed Data)

```bash
jupyter notebook reproduce.ipynb
```

Run cells sequentially to reproduce all key numbers reported in the paper.

### Running Individual Scripts

```bash
python code/fig_error_vs_attack.py      # Generates figures/fig2.png
python code/statistical_analysis.py     # Generates stat_*.csv in data/
python code/threshold_sensitivity.py    # Prints Table S4 results
```

## Script-to-Paper Mapping

| Script                               | Output            | Paper Location               |
| ------------------------------------ | ----------------- | ---------------------------- |
| **Core pipeline**                    |                   |                              |
| `pipeline/batch_runner.py`           | match_summary.csv | Data source for entire paper |
| **Main text figures & tables**       |                   |                              |
| `fig_multiscale.py`                  | fig1.png/pdf      | §2.1 Fig 1                   |
| `fig_error_vs_attack.py`             | fig2.png/pdf      | §3.1 Fig 2                   |
| `fig_structural_factors.py`          | fig3.png/pdf      | §3.2 Fig 3                   |
| `fig_partial_correlation.py`         | fig4.png/pdf      | §3.2 Fig 4                   |
| `statistical_analysis.py`            | stat_*.csv        | §3.2 Table 2                 |
| `fig_mr_benchmark.py`               | fig5.png/pdf      | §3.3 Fig 5                   |
| `fig_rai_scatter.py`                 | fig6.png/pdf      | §3.4 Fig 6                   |
| `fig_team_stability.py`             | fig7.png/pdf      | §3.5 Fig 7                   |
| `fig_baena_case.py`                  | fig8.png/pdf      | §3.5 Fig 8                   |
| `cross_sectional_validation.py`      | console output    | §3.5 Table 4                 |
| **Supplementary material**           |                   |                              |
| `fig_corr_heatmap.py`               | figS1.png/pdf     | SM S2 Fig S1                  |
| `fig_position_analysis.py`           | figS2.png/pdf     | SM S3 Fig S2                  |
| `threshold_sensitivity.py`           | console output    | SM S5 Table S4                |
| `scgc_vs_wgcc.py`                    | console output    | SM S6 Table S5                |

## Data Sources

- **Passing network data**: StatsBomb (https://statsbomb.com/)
  - English Premier League (EPL) + La Liga, seasons 2022/23, 2023/24, 2024/25
  - 2,280 matches × 2 teams per match = 4,560 single-match networks
  - 4 spatial scales (L0–L3), yielding 18,240 networks in total
- **Player market valuations**: Transfermarkt (https://www.transfermarkt.com)

## Notes

- `data/match_summary.csv` contains all pre-computed results; most scripts run directly without raw data
- `fig_multiscale.py` and `fig_baena_case.py` additionally require raw StatsBomb event JSONs and the `pipeline/` modules
- `pipeline/batch_runner.py` requires raw StatsBomb data; configure `DATA_ROOT` before running
- Percolation simulation parameters: 20 Monte Carlo trials for L2 (primary analysis), 5 for L0/L1/L3 (supporting)
