# Scale-dependent applicability of directed mean-field percolation in soccer passing networks

Code and data for reproducing the results in

> Huang K, Zhou C, Xiong X, Yang X, Song Y, Xu K, Martínez JH, Gómez-Ruano MÁ, Buldú JM.
> *Scale-dependent applicability of directed mean-field percolation in soccer passing networks.*
> Chaos, Solitons & Fractals 213 (2026) 119356. https://doi.org/10.1016/j.chaos.2026.119356

The repository contains the match-level results for all 18,240 networks of the article (2,280 matches of the English Premier League and La Liga, 2022/23 to 2024/25, two teams per match, four spatial resolutions), one script for every figure, table and number of the article and its Supplementary Material, and the pipeline that builds the networks and runs the percolation simulations from the raw StatsBomb event data.

Every script that reproduces published values compares them with the article and exits with status 1 if any value differs, printing what differs. Most scripts run on the shipped files in a few seconds. Scripts marked † need the raw event data, which StatsBomb licenses and which cannot be redistributed here.

## Quick start

```bash
# Python 3.10 with pandas, numpy, scipy, matplotlib, seaborn, statsmodels,
# scikit-learn, networkx, pingouin 0.5.x and mplsoccer
python code/results_numbers_table2.py      # every number of the Results section and Table 2
python code/fig3_error_vs_attack.py        # writes figures/fig3.pdf and figures/fig3.png
jupyter notebook reproduce.ipynb           # runs all scripts in order
```

Run the scripts from the repository root. Figures are written to `figures/` and tables to `results/`; both folders are created on first use. pingouin 0.6 renames the ICC types and is not supported; use the 0.5 series.

## Repository structure

```
passing_network_robustness/
├── README.md
├── CITATION.cff
├── reproduce.ipynb                     # runs every script and shows the checks
├── data/                               # shipped results and inputs (see Data)
└── code/
    ├── fig2_multiscale.py ... fig9_baena_case.py   # main-text figures
    ├── results_numbers_table2.py       # Results section numbers and Table 2
    ├── table4_market_value.py          # Table 4
    ├── in_out_degree_correlation.py    # in- and out-degree correlation, Section 3.3
    ├── synthetic_baselines_generator.py  # synthetic reference networks, Section 3.4
    ├── style.py                        # plotting style shared by Figs. 3 to 5
    ├── supplementary/                  # one or more scripts per Supplementary section S1 to S14
    └── pipeline/                       # raw events to networks to percolation results
```

## Main text

| Item | Script | Input |
| --- | --- | --- |
| Fig. 1 | workflow diagram, drawn by hand | |
| Fig. 2 | `fig2_multiscale.py` † | raw events of Manchester City vs Burnley, 2023/24 |
| Fig. 3 | `fig3_error_vs_attack.py` | `match_summary_mc100_guarded.csv` |
| Fig. 4 | `fig4_structural_factors.py` | same |
| Fig. 5 | `fig5_partial_correlation.py` | same |
| Fig. 6 and Table 3 | `fig6_table3_mr_benchmark.py` | same |
| Fig. 7 | `fig7_rai_scatter.py` | same, `synthetic_mc100_guarded.csv` |
| Fig. 8 | `fig8_team_stability.py` | `match_summary_mc100_guarded.csv` |
| Fig. 9 | `fig9_baena_case.py` † | raw events of Villarreal vs Girona, 2024/25 |
| Table 1 | definitions, no computation | |
| Table 2 and the numbers in Sections 3.1 to 3.5 | `results_numbers_table2.py` | `match_summary_mc100_guarded.csv`, `synthetic_mc100_guarded.csv` |
| Table 4 and the player-level numbers in Section 3.5 | `table4_market_value.py` | `player_impact_*.csv`, `transfermarkt_*.csv`, `market_value_validation_all_seasons.csv` |
| In- and out-degree correlation by scale, Section 3.3 | `in_out_degree_correlation.py` (full run †) | `supplementary/in_out_degree_correlation.csv` |
| Synthetic reference networks, Section 3.4 | `synthetic_baselines_generator.py` | `match_summary_mc100_guarded.csv` (calibration) |

## Supplementary Material

All scripts are in `code/supplementary/`.

| Section | Item | Script | Input |
| --- | --- | --- | --- |
| S1 | Table S1, rank consistency across scales | `s01_descriptive.py` | `match_summary_mc100_guarded.csv` |
| S2 | Fig. S1 | `s02_corr_heatmap.py` | `match_summary.csv`, `match_summary_mc100_guarded.csv` |
| S3 | Tables S2 and S3, Fig. S2 | `s03_position_mediation.py` | `position_node_records.csv`, `position_network_records.csv`, `match_summary_mc100_guarded.csv` |
| S4 | substitute-inheritance check | `s04_substitute_inheritance.py` (build †) | `player_impact_*.csv`, `supplementary/player_impact_*_inherit.csv`, `supplementary/s4_identities_per_match.csv` |
| S5 | Tables S4 and S5 | `s05_threshold_sensitivity.py` | `match_summary_mc100_guarded.csv` |
| S6 | Table S6 | `s06_scgc_vs_wgcc.py` | same |
| S7 | Fig. S3 | `s07_convergence.py` | `supplementary/convergence_results.csv` |
| S8 | Table S7 | `s08_calibration.py` | none (synthetic ER graphs) |
| S8 | Fig. S4 | `s08_example_curves.py` † | raw events of six networks |
| S8 | Table S8 | `s08_threshold_settings.py` | `match_summary.csv`, `match_summary_mc100_guarded.csv` |
| S9 | Fig. S5 | `s09_within_scale.py` | `match_summary_mc100_guarded.csv` |
| S10 | Fig. S6 | `s10_rewiring_null.py --plot` (full run †) | `supplementary/summary_by_scale.csv`, `supplementary/structure_summary.csv` |
| S11 | Fig. S7 | `s11_degree_distributions.py` | `supplementary/degrees_*.csv` |
| S12 | Fig. S8 | `s12_pca.py` | `match_summary_mc100_guarded.csv` |
| S13 | Fig. S9 | `s13_fc_r_integral.py` | same |
| S14 | Fig. S10 | `s14_ebc_ranking.py --plot` (full run †) | `supplementary/ebc_vs_degree_all_matches.csv`, `match_summary.csv` |

## Data

| File | Content |
| --- | --- |
| `match_summary_mc100_guarded.csv` | Results of the article. One row per team-match network and resolution (18,240 rows), with the network size, the directed structural metrics, the SCGC and WGCC thresholds and robustness integrals under random failure and targeted attack, and the directed MR threshold of Eq. (1). |
| `match_summary.csv` | The same networks evaluated with the first setting of Supplementary Table S8 (20 random-failure trials at L2 and 5 at L0, L1 and L3, without the order-parameter restriction). Used for Table S8, Fig. S1 and Section S14. Structural metrics are identical to the file above. |
| `synthetic_mc100_guarded.csv` | 100 realisations each of the calibrated ER, BA, WS and RGG reference networks evaluated with the protocol of the article. |
| `player_impact_2022_23.csv`, `player_impact_202324.csv`, `player_impact_202425.csv` | Player-level SCGC edge-sensitivity scores per season (Section 3.5). |
| `transfermarkt_players.csv`, `transfermarkt_player_valuations.csv`, `market_value_validation_all_seasons.csv` | Transfermarkt players and valuations and the matching between StatsBomb and Transfermarkt names (Table 4). |
| `position_node_records.csv`, `position_network_records.csv` | Node-level degrees with the positional role and network-level positional summaries at L2 (Section S3). |
| `supplementary/` | Inputs of the supplementary scripts that would otherwise need the raw data or long simulations. |

Main columns of `match_summary_mc100_guarded.csv`

| Column | Meaning |
| --- | --- |
| `League`, `Season`, `Team`, `TeamID`, `Venue`, `MatchID`, `Opponent`, `Scale` | identifiers; `Scale` is L0 (24 × 16 zones), L1 (12 × 8), L2 (6 × 4) or L3 (3 × 2) |
| `PassCount`, `UniquePlayers`, `Nodes`, `Edges` | completed open-play passes, players, nodes and directed edges |
| `AvgDegree`, `Clustering`, `Assortativity`, `Reciprocity`, `FlowHierarchy`, `OutDegreeCV`, `Mu11_Mu`, `CyclicTriadFraction` | the eight directed structural metrics |
| `SCGC_Ratio`, `WGCC_Ratio` | initial size of the strongly and weakly connected giant component relative to the number of nodes |
| `pc_Error_Strong`, `pc_Attack_Strong` | pseudo-critical threshold of the SCGC under random failure and targeted attack |
| `pc_Error_Weak`, `pc_Attack_Weak` | the same for the WGCC |
| `R_*`, `ChiMax_*` | robustness integral and susceptibility maximum for each protocol and order parameter |
| `pc_Theory`, `TheoryDeviation` | directed MR threshold of Eq. (1) and its difference from `pc_Error_Strong` |
| `P0_strong`, `fcE_raw_strong`, `fcA_raw_strong`, `PinfAtRawPeakE_strong`, `GuardFloor_strong` | initial SCGC fraction, unrestricted Error and Attack thresholds with 100 trials (second setting of Table S8), P∞ at the unrestricted Error peak, and the restriction floor 0.05 × P∞(0) |

The primary L2 sample of the article (N = 4,330) consists of the L2 rows with `SCGC_Ratio >= 0.3`, `pc_Error_Strong > 0` and `pc_Attack_Strong > 0`.

In `synthetic_mc100_guarded.csv`, `fcE_guarded` and `fcA_guarded` are the thresholds with the restricted estimator, `fcE_raw` and `fcA_raw` without it, `pc_Theory` the directed MR threshold of Eq. (1), and `RAI` the robustness asymmetry index of the realisation.

## Percolation protocol

| Parameter | Value |
| --- | --- |
| Random failure (Error) | 100 Monte Carlo trials at all four resolutions, seed 42 |
| Targeted attack (Attack) | single deterministic removal, edges ranked once by the sum of the total degrees of their endpoints |
| Removal levels | 50 equally spaced levels, f = k/49 |
| Order parameter | P∞(f), size of the strongly connected giant component relative to N |
| Susceptibility | χ(f) = N⁻¹ Σ cᵢ² over the non-largest strongly connected components |
| Trial averaging | P∞ and χ averaged over trials at each removal level before the maximum is located |
| Threshold | f_c = argmax χ(f), searched only where P∞(f) ≥ 0.05 P∞(0); the same restriction is used for both protocols and both order parameters |

The restriction excludes spurious late maxima of χ after the giant component has vanished. Its calibration on directed ER graphs with known thresholds is reproduced by `s08_calibration.py` (Table S7).

## Rebuilding the results from the raw event data

```bash
export DATA_ROOT=/path/to/statsbomb/datos    # folders {competition_id}_{season_id}_*/{match_id}_events.json
python code/pipeline/batch_runner.py --test  # one team of one season, about two minutes
python code/pipeline/batch_runner.py         # all 18,240 networks, several hours
```

The pipeline (`code/pipeline/`) parses the events (`data_loader.py`), builds the pitch-player networks at the four resolutions (`network_builder.py`), computes the structural metrics (`compute_directed_indicators.py`) and runs the percolation simulations (`robustness.py`). `batch_runner.py` writes `results/match_summary.csv`, which equals the shipped `data/match_summary_mc100_guarded.csv` except for the names of the threshold columns and the five diagnostic columns listed above.

| `results/match_summary.csv` | `data/match_summary_mc100_guarded.csv` |
| --- | --- |
| `fc_Error_Strong`, `fc_Error_Weak`, `fc_Attack_Strong`, `fc_Attack_Weak` | `pc_Error_Strong`, `pc_Error_Weak`, `pc_Attack_Strong`, `pc_Attack_Weak` |
| `fc_dMR` | `pc_Theory` |
| `dMR_Deviation` | `TheoryDeviation` |

All other columns have the same names and values.

## Data sources

- Event data from StatsBomb (https://statsbomb.com). English Premier League and La Liga, seasons 2022/23, 2023/24 and 2024/25. The raw data are not redistributed; only derived results are included.
- Player market values from Transfermarkt (https://www.transfermarkt.com).

## Citation

If you use this code or data, please cite

```bibtex
@article{Huang2026CSF,
  author  = {Huang, Kongyun and Zhou, Changjing and Xiong, Xiuyuan and Yang, Xudong and Song, Yingzhe and Xu, Kai and Mart{\'i}nez, Johann H. and G{\'o}mez-Ruano, Miguel-{\'A}ngel and Buld{\'u}, Javier M.},
  title   = {Scale-dependent applicability of directed mean-field percolation in soccer passing networks},
  journal = {Chaos, Solitons \& Fractals},
  volume  = {213},
  pages   = {119356},
  year    = {2026},
  doi     = {10.1016/j.chaos.2026.119356}
}
```

## License

MIT, see `LICENSE`.
