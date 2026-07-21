# Scale-dependent Applicability of Directed Mean-field Percolation in Soccer Passing Networks

Code and data for reproducing the results in the paper.

## Revision Update (July 2026)

This repository was updated alongside the revised manuscript. Summary of changes:

- **Percolation protocol revised**: 100 Monte Carlo trials at all four scales (previously 20 for L2 and 5 for L0/L1/L3), susceptibility curves averaged pointwise across trials before locating the maximum, and a peak-validity restriction on the susceptibility estimator (see "Percolation protocol" below). `code/pipeline/` implements the revised protocol.
- **New datasets**: `data/match_summary_mc100_guarded.csv` and `data/synthetic_mc100_guarded.csv` contain the revised-protocol results used by the revised manuscript. The original files (`match_summary.csv`, `synthetic_baselines.csv`) are kept unchanged for provenance.
- **New supplementary analysis scripts**: `code/supplementary/` contains one script per revised supplementary section S7–S14, plus supporting input data under `data/supplementary/`.
- **Synthetic generator released**: `code/synthetic_baselines_generator.py` generates the calibrated ER/BA/WS/RGG baselines from scratch and evaluates them under the revised protocol.
- **Figure files removed**: pre-generated figures are no longer shipped (the manuscript's figure numbering changed in revision); all figures can be regenerated from the scripts (see mapping below).

## Project Structure

```
passing_network_robustness/
├── README.md
├── reproduce.ipynb                    # Reproducibility notebook
├── data/                              # Pre-computed data files
│   ├── match_summary.csv              # Primary dataset, original submission protocol (18,240 rows)
│   ├── match_summary_mc100_guarded.csv# Primary dataset, revised protocol (18,240 rows + 5 audit columns)
│   ├── synthetic_baselines.csv        # Synthetic baselines, original protocol (ER/BA/WS/RGG)
│   ├── synthetic_mc100_guarded.csv    # Synthetic baselines, revised protocol
│   ├── batch_summary.csv              # Team–season–venue–scale arithmetic means (960 rows)
│   ├── player_impact_*.csv            # Player structural impact (3 seasons)
│   ├── market_value_validation_all_seasons.csv  # Transfermarkt value matching
│   ├── transfermarkt_*.csv            # Transfermarkt player valuations
│   ├── position_node_records.csv      # Node-level positional degree data
│   ├── position_network_records.csv   # Network-level positional degree data
│   ├── stat_*.csv                     # §3.2 statistics, original submission protocol (N = 4,343)
│   └── supplementary/                 # Inputs for the supplementary scripts (see S7–S14 table)
└── code/
    ├── pipeline/                      # Core data-generation pipeline (JSON → match_summary.csv)
    │   ├── batch_runner.py            #   Orchestrator: iterates over all matches (revised protocol)
    │   ├── data_loader.py             #   StatsBomb JSON parsing, pass extraction, substitution inheritance
    │   ├── network_builder.py         #   Multi-scale position-dependent directed network construction
    │   ├── robustness.py              #   Bond percolation Monte Carlo + restricted threshold estimator
    │   └── compute_directed_indicators.py  # Directed structural indicators
    ├── supplementary/                 # Revised supplementary analyses S7–S14 (one script per section)
    ├── synthetic_baselines_generator.py  # Calibrated synthetic baselines, revised protocol
    ├── fig_multiscale.py              # → Fig 2
    ├── fig_error_vs_attack.py         # → Fig 3
    ├── fig_structural_factors.py      # → Fig 4
    ├── fig_partial_correlation.py     # → Fig 5
    ├── fig_mr_benchmark.py            # → Fig 6
    ├── fig_rai_scatter.py             # → Fig 7
    ├── fig_team_stability.py          # → Fig 8
    ├── fig_baena_case.py              # → Fig 9
    ├── fig_corr_heatmap.py            # → Fig S1 (SM S2)
    ├── fig_position_analysis.py       # → Fig S2 (SM S3)
    ├── statistical_analysis.py        # → stat_*.csv
    ├── cross_sectional_validation.py  # → Table 4
    ├── threshold_sensitivity.py       # → SM S5
    └── scgc_vs_wgcc.py               # → SM S6
```

Note: Figure 1 of the revised manuscript is a workflow diagram and is not a pipeline product. The former Figures 1–8 of the original submission are Figures 2–9 in the revision; the script names are unchanged.

## Percolation Protocol (Revised Manuscript)

| Parameter | Value |
| --- | --- |
| Monte Carlo trials (random removal, "Error") | B = 100 at all four scales |
| Targeted removal ("Attack") | Single deterministic trajectory (static edge ranking by endpoint degree sum) |
| Removal fractions | 50 equally spaced levels (f-grid resolution 1/49 ≈ 0.020) |
| Susceptibility | χ(f) = N⁻¹ Σ cᵢ² over non-largest strongly connected components |
| Trial averaging | P∞ and χ curves averaged pointwise across trials **before** locating the peak |
| Threshold estimator | f_c = argmax χ(f), restricted to f where P∞(f) ≥ 5% of P∞(0) |
| Restriction scope | Applied identically to Error and Attack, and to SCGC and WGCC order parameters |
| Random seed | 42 (fixed; results are bit-reproducible) |
| Validity criteria (L2 primary analysis) | Initial SCGC ≥ 30% of nodes AND f_c(Error) > 0 → N = 4,330 |

The peak-validity restriction suppresses spurious late-stage χ maxima that occur after the strongly connected core has effectively collapsed. Its calibration on directed ER graphs with an exactly known threshold is reproduced by `code/supplementary/s08_calibration.py`.

## Data-Generation Pipeline

`code/pipeline/` contains the complete code for generating the match-level dataset from raw StatsBomb event data, implementing the revised protocol.

### Pipeline Flow

```
StatsBomb event data (2 leagues × 3 seasons × ~380 matches)
    ↓  data_loader.py: parse JSON, extract successful open-play passes, handle substitution inheritance
    ↓  network_builder.py: build position-dependent directed passing networks at 4 scales (L0–L3)
    ↓  compute_directed_indicators.py: compute reciprocity, flow hierarchy, triadic census, etc.
    ↓  robustness.py: bond percolation Monte Carlo (100 trials, restricted estimator) + MR analytical threshold
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
python batch_runner.py          # Full run (revised protocol; several hours)
python batch_runner.py --test   # Test mode (EPL 22/23, first 50 matches)
```

**Note**: `batch_runner.py` now implements the revised protocol and writes to `data/match_summary.csv`. Re-running it therefore **overwrites the shipped original-protocol file with revised-protocol values** (with `fc_*` column names and no audit columns). The shipped files are provided precisely so that no rerun is needed: `match_summary.csv` preserves the original submission protocol, and `match_summary_mc100_guarded.csv` is the revised-protocol production output used by the revised manuscript.

## Two Protocol Datasets

| File | Protocol | Columns |
| --- | --- | --- |
| `data/match_summary.csv` | Original submission: 20 trials (L2) / 5 (L0/L1/L3), unrestricted estimator | 49, `fc_*` naming |
| `data/match_summary_mc100_guarded.csv` | Revised: 100 trials all scales, restricted estimator | 49 + 5 audit columns, `pc_*` naming |
| `data/synthetic_baselines.csv` | Original submission (5 trials, unrestricted) | `fc_*` naming |
| `data/synthetic_mc100_guarded.csv` | Revised (100 trials + restriction), with regeneration gate flags | see below |

The revised-protocol files are byte-identical copies of the production outputs and retain the production column names. Correspondence between the two naming schemes:

| `match_summary.csv` | `match_summary_mc100_guarded.csv` |
| --- | --- |
| `fc_Error_Strong` / `fc_Error_Weak` | `pc_Error_Strong` / `pc_Error_Weak` |
| `fc_Attack_Strong` / `fc_Attack_Weak` | `pc_Attack_Strong` / `pc_Attack_Weak` |
| `fc_dMR` | `pc_Theory` |
| `dMR_Deviation` | `TheoryDeviation` |
| (all other 43 columns) | identical names |

Of the 43 shared column names, 37 (identifiers and structural metrics) hold identical values in both files; the six percolation-derived columns `R_Error_Strong`, `R_Error_Weak`, and `ChiMax_{Error,Attack}_{Strong,Weak}` hold original-protocol values in `match_summary.csv` and revised-protocol values in the guarded file. Among the renamed columns, the four thresholds and the deviation column likewise hold revised-protocol values in the guarded file, whereas the analytical threshold (`fc_dMR` = `pc_Theory`) is protocol-independent and identical in both.

The five audit columns of the revised file record, per network: the initial SCGC fraction (`P0_strong`), the unrestricted Error and Attack thresholds (`fcE_raw_strong`, `fcA_raw_strong`), the P∞ value at the unrestricted Error peak (`PinfAtRawPeakE_strong`), and the restriction floor (`GuardFloor_strong` = 0.05 × P0).

`synthetic_mc100_guarded.csv` columns: per realisation, structural summaries plus restricted (`fcE_guarded`, `fcA_guarded`) and unrestricted (`fcE_raw`, `fcA_raw`) thresholds, robustness integrals, benchmark readings, RAI, and two regeneration gates (`gateA`: regenerated graph bit-identical to the archived one; `gateB`: original protocol reproduces the archived thresholds).

Headline revised-protocol numbers (L2, N = 4,330): f_c(Error) = 0.604 (SD 0.138), f_c(Attack) = 0.420 (SD 0.141), 95.1% of matches more robust to random than targeted removal; directed MR benchmark 0.601 vs 0.601, Bias = −0.001, r = 0.809, MAE = 0.064. These are verified in Cell 1c of `reproduce.ipynb`.

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

Run cells sequentially. Cell 1c verifies the revised-protocol headline numbers; the remaining cells verify the originally submitted analysis from the original-protocol files.

### Running Individual Scripts

```bash
python code/fig_error_vs_attack.py             # Generates figures/fig2.png (paper Fig 3)
python code/statistical_analysis.py            # Generates stat_*.csv in data/
python code/supplementary/s12_pca.py           # SM S12 PCA (gates against published values)
python code/synthetic_baselines_generator.py --test   # Synthetic baselines, 5 realisations/model
```

## Script-to-Paper Mapping (Main Text)

| Script | Paper location (revised numbering) |
| --- | --- |
| **Core pipeline** | |
| `pipeline/batch_runner.py` | Data source for the entire paper |
| **Figures & tables** | |
| — (workflow diagram, not a pipeline product) | Fig 1 |
| `fig_multiscale.py` † | Fig 2 |
| `fig_error_vs_attack.py` | Fig 3 |
| `fig_structural_factors.py` | Fig 4 |
| `fig_partial_correlation.py` | Fig 5 |
| `fig_mr_benchmark.py` | Fig 6 |
| `fig_rai_scatter.py` | Fig 7 |
| `fig_team_stability.py` | Fig 8 |
| `fig_baena_case.py` † | Fig 9 |
| `statistical_analysis.py` | §3.2 Table 2 |
| `cross_sectional_validation.py` | §3.5 Table 4 |

† requires raw StatsBomb event JSONs in addition to the shipped CSVs.

The `fig_*.py` scripts read the original-protocol CSVs and therefore reproduce the originally submitted versions of the figures; the revised manuscript's figures additionally reflect the revised-protocol values (`match_summary_mc100_guarded.csv`).

## Supplementary Material Mapping

| SM section | Script | Input data |
| --- | --- | --- |
| S1 Multi-scale network characteristics | (descriptive; `match_summary.csv` aggregates) | `data/` |
| S2 Inter-metric correlation structure (Fig S1) | `fig_corr_heatmap.py` | `data/match_summary.csv` |
| S3 Positional degree analysis (Fig S2) | `fig_position_analysis.py` | `data/position_*.csv` |
| S4 Substitute-inheritance robustness check | (player-level residual comparison; no dedicated script in this repository) | — |
| S5 SCGC threshold sensitivity | `threshold_sensitivity.py` | `data/match_summary.csv` |
| S6 SCGC vs WGCC order parameters | `scgc_vs_wgcc.py` | `data/match_summary.csv` |
| S7 Monte Carlo convergence (Fig S3) | `supplementary/s07_convergence.py` | `data/supplementary/convergence_results.csv` |
| S8 Order-parameter restriction / ER calibration (Table S7) | `supplementary/s08_calibration.py` | none (synthetic) |
| S9 Within-scale network-size check (Fig S5) | `supplementary/s09_within_scale.py` | `data/match_summary_mc100_guarded.csv` |
| S10 Degree-preserving rewiring null model (Fig S6) | `supplementary/s10_rewiring_null.py` | guarded CSV + raw data (full run); `data/supplementary/` (plot mode) |
| S11 Degree distributions (Fig S7) | `supplementary/s11_degree_distributions.py` | `data/supplementary/degrees_*.csv`, `ks_distances.csv` |
| S12 PCA of structural metrics (Fig S8) | `supplementary/s12_pca.py` | `data/match_summary_mc100_guarded.csv` |
| S13 Threshold vs robustness integral (Fig S9) | `supplementary/s13_fc_r_integral.py` | `data/match_summary_mc100_guarded.csv` |
| S14 EBC vs degree-based attack (Fig S10) | `supplementary/s14_ebc_ranking.py` | raw data (full run); `data/supplementary/ebc_vs_degree_all_matches.csv` (plot mode) |

Several supplementary scripts embed acceptance gates that re-verify the published numbers from the shipped data and exit on mismatch.

## Data Sources

- **Passing network data**: StatsBomb (https://statsbomb.com/)
  - English Premier League (EPL) + La Liga, seasons 2022/23, 2023/24, 2024/25
  - 2,280 matches × 2 teams per match = 4,560 single-match networks
  - 4 spatial scales (L0–L3), yielding 18,240 networks in total
  - Raw event data are not redistributed here; only derived aggregate summaries are included
- **Player market valuations**: Transfermarkt (https://www.transfermarkt.com)

## Notes

- `data/match_summary.csv` and `data/match_summary_mc100_guarded.csv` contain all pre-computed match-level results; most scripts run directly without raw data
- `fig_multiscale.py`, `fig_baena_case.py`, the full runs of `supplementary/s10_rewiring_null.py` and `supplementary/s14_ebc_ranking.py`, and `pipeline/batch_runner.py` require raw StatsBomb data (configure `DATA_ROOT`)
- The 14 analysis scripts directly under `code/` read the original-protocol CSVs and reproduce the originally submitted analysis; the revised-protocol analyses are covered by `code/pipeline/`, `code/supplementary/`, `code/synthetic_baselines_generator.py`, and Cell 1c of the notebook
- `data/stat_*.csv` retain the original submission values (N = 4,343); the corresponding revised-protocol statistics (N = 4,330) are re-verified by the acceptance gates in `supplementary/s12_pca.py`
- Revised percolation parameters: 100 Monte Carlo trials at all scales, peak-validity restriction at 5% of the initial order parameter, 50 removal levels, seed 42
