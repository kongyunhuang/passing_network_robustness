"""
pipeline — Football passing network data generation pipeline
=============================================================
Starting from StatsBomb JSON raw event data, the following steps produce the
match-level results (shipped as data/match_summary_mc100_guarded.csv):

    StatsBomb JSON -> data_loader (parsing) -> network_builder (network construction)
    -> robustness (percolation simulation) + compute_directed_indicators (directed metrics)
    -> batch_runner (orchestration) -> results/match_summary.csv, results/batch_summary.csv

Module list:
    data_loader                 StatsBomb JSON event parsing and pass extraction
    network_builder             Multi-scale position-dependent directed passing network construction
    robustness                  Bond percolation simulation and robustness computation
    compute_directed_indicators Directed network structural indicator computation
    batch_runner                Batch analysis orchestrator (main entry point)

Last modified: 2026-10-27
"""
