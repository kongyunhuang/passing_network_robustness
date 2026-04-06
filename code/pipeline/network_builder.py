"""
network_builder.py — Multi-scale position-dependent directed passing network construction
=========================================================================================
Divides the pitch into zones at multiple granularities (L0-L3) and builds
position-dependent directed passing networks. Nodes are (player_id, zone_id)
tuples; directed edges represent passing relationships (MultiDiGraph).

Input:  List of pass records (from data_loader.extract_passes)
Output: NetworkX MultiDiGraph with node attributes and network statistics

Scale definitions (StatsBomb coordinate system 120x80 yards):
    L0: 24x16 = 384 zones (5x5 yards)
    L1: 12x8  = 96 zones  (10x10 yards)
    L2: 6x4   = 24 zones  (20x20 yards, primary analysis scale)
    L3: 3x2   = 6 zones   (40x40 yards)

Usage:
    from pipeline.network_builder import ZonalNetworkBuilder
    builder = ZonalNetworkBuilder()
    G = builder.build_network(passes, scale='L2')
    stats = builder.get_network_stats(G)

Last modified: 2026-03-27
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple


# ============== Multi-scale definitions ==============

# Zone granularity configuration (columns, rows)
# StatsBomb coordinate system: 120 x 80 yards
SCALE_CONFIGS = {
    'L0': (24, 16),   # 384 zones, ultra-fine (5x5 yards per zone)
    'L1': (12, 8),    # 96 zones, fine (10x10 yards per zone)
    'L2': (6, 4),     # 24 zones, medium (20x20 yards, Ichinose level)
    'L3': (3, 2),     # 6 zones, coarse (40x40 yards, coarsest macro)
}


class ZonalNetworkBuilder:
    """
    Multi-scale zonal passing network builder.

    Usage:
        builder = ZonalNetworkBuilder()
        G = builder.build_network(passes, scale='L1')
        rg_flow = builder.build_rg_flow(passes)
    """

    def __init__(self, field_length: float = 120.0, field_width: float = 80.0):
        """
        Initialise the builder.

        Args:
            field_length: Pitch length (StatsBomb default 120 yards)
            field_width: Pitch width (StatsBomb default 80 yards)
        """
        self.field_length = field_length
        self.field_width = field_width

    def get_zone_id(self, x: float, y: float, n_cols: int, n_rows: int) -> int:
        """
        Map coordinates to a zone ID.

        Formula: zone_id = floor(x/L * n_cols) + floor(y/W * n_rows) * n_cols

        Args:
            x: X coordinate (0-120)
            y: Y coordinate (0-80)
            n_cols: Number of columns (X-axis divisions)
            n_rows: Number of rows (Y-axis divisions)

        Returns:
            zone_id: Integer in range [0, n_cols * n_rows - 1]
        """
        # Boundary protection to prevent coordinates exactly at boundary values
        x = max(0, min(x, self.field_length - 0.001))
        y = max(0, min(y, self.field_width - 0.001))

        col = int(x / self.field_length * n_cols)
        row = int(y / self.field_width * n_rows)

        return row * n_cols + col

    def get_zone_center(self, zone_id: int, n_cols: int, n_rows: int) -> Tuple[float, float]:
        """
        Get the centre coordinates of a zone (for visualisation).

        Args:
            zone_id: Zone ID
            n_cols, n_rows: Zone division parameters

        Returns:
            (x_center, y_center): Zone centre coordinates
        """
        col = zone_id % n_cols
        row = zone_id // n_cols

        zone_width = self.field_length / n_cols
        zone_height = self.field_width / n_rows

        x_center = col * zone_width + zone_width / 2
        y_center = row * zone_height + zone_height / 2

        return x_center, y_center

    def build_network(
        self,
        passes: List[Dict],
        scale: str = 'L1'
    ) -> nx.MultiDiGraph:
        """
        Build a position-dependent directed passing network from pass data.

        Args:
            passes: List of pass records (from data_loader.extract_passes).
                    Each element contains: player_id, recipient_id,
                    x_start, y_start, x_end, y_end
            scale: Granularity level ('L0', 'L1', 'L2', 'L3')

        Returns:
            G: NetworkX MultiDiGraph
                - Nodes: (player_id, zone_id) tuples
                - Directed edges: A->B indicates a pass from A to B (multi-edges allowed)
                - Node attributes: player_name, zone_id, zone_center
        """
        if scale not in SCALE_CONFIGS:
            raise ValueError(f"Invalid scale level: {scale}. Options: {list(SCALE_CONFIGS.keys())}")

        n_cols, n_rows = SCALE_CONFIGS[scale]
        G = nx.MultiDiGraph()

        # Store additional node information
        node_info = {}

        for p in passes:
            # Get start and end zones
            zone_start = self.get_zone_id(
                p['x_start'], p['y_start'], n_cols, n_rows
            )
            zone_end = self.get_zone_id(
                p['x_end'], p['y_end'], n_cols, n_rows
            )

            # Build node identifiers: (player_id, zone_id)
            node_start = (p['player_id'], zone_start)
            node_end = (p['recipient_id'], zone_end)

            # Record node information
            if node_start not in node_info:
                node_info[node_start] = {
                    'player_id': p['player_id'],
                    'player_name': p.get('player_name', 'Unknown'),
                    'zone_id': zone_start,
                    'zone_center': self.get_zone_center(zone_start, n_cols, n_rows)
                }
            if node_end not in node_info:
                node_info[node_end] = {
                    'player_id': p['recipient_id'],
                    'player_name': p.get('recipient_name', 'Unknown'),
                    'zone_id': zone_end,
                    'zone_center': self.get_zone_center(zone_end, n_cols, n_rows)
                }

            # Add edge (MultiDiGraph allows parallel edges)
            G.add_edge(node_start, node_end)

        # Set node attributes
        nx.set_node_attributes(G, node_info)

        # Store network metadata
        G.graph['scale'] = scale
        G.graph['n_cols'] = n_cols
        G.graph['n_rows'] = n_rows
        G.graph['n_zones'] = n_cols * n_rows

        return G

    def build_rg_flow(self, passes: List[Dict]) -> Dict[str, nx.MultiDiGraph]:
        """
        Build a renormalisation group flow from L0 to L3.

        Args:
            passes: List of pass records

        Returns:
            rg_flow: {scale_name: network} dictionary
        """
        return {
            scale: self.build_network(passes, scale)
            for scale in SCALE_CONFIGS.keys()
        }

    def get_network_stats(self, G) -> Dict:
        """
        Compute basic network statistics (supports both directed and undirected graphs).

        Args:
            G: Network graph (MultiDiGraph, DiGraph, MultiGraph, or Graph)

        Returns:
            stats: Dictionary containing various statistics
        """
        # Convert to simple graph (directed or undirected)
        if G.is_directed():
            G_simple_dir = nx.DiGraph(G)
            G_simple_undir = nx.Graph(G)  # For metrics requiring undirected graph (e.g. clustering)
        else:
            G_simple_dir = None
            G_simple_undir = nx.Graph(G)

        n_nodes = G.number_of_nodes()
        unique_players = set()
        for node, attrs in G.nodes(data=True):
            player_id = attrs.get('player_id')
            if player_id is None and isinstance(node, tuple) and node:
                player_id = node[0]
            if player_id is not None:
                unique_players.add(player_id)

        G_for_degree = G_simple_dir if G_simple_dir else G_simple_undir
        stats = {
            'scale': G.graph.get('scale', 'Unknown'),
            'n_nodes': n_nodes,
            'n_unique_players': len(unique_players),
            'n_edges': G.number_of_edges(),
            'n_unique_edges': G_simple_dir.number_of_edges() if G_simple_dir else G_simple_undir.number_of_edges(),
            # Directed graph: <k> = E/N (= <k_in> = <k_out>); Undirected graph: 2E/N
            'avg_degree': (G_for_degree.number_of_edges() / n_nodes) if n_nodes > 0 else 0,
            'density': nx.density(G_for_degree) if n_nodes > 1 else 0,
        }

        # Directed-graph-specific metrics
        if G_simple_dir and n_nodes > 0:
            in_degrees = [d for n, d in G_simple_dir.in_degree()]
            out_degrees = [d for n, d in G_simple_dir.out_degree()]
            stats['avg_in_degree'] = np.mean(in_degrees)
            stats['avg_out_degree'] = np.mean(out_degrees)

            # Strongly connected components
            scc = list(nx.strongly_connected_components(G_simple_dir))
            largest_scc = max(scc, key=len) if scc else set()
            stats['n_strongly_connected'] = len(scc)
            stats['largest_scc_size'] = len(largest_scc)
            stats['largest_scc_ratio'] = len(largest_scc) / n_nodes

            # Weakly connected components
            wcc = list(nx.weakly_connected_components(G_simple_dir))
            largest_wcc = max(wcc, key=len) if wcc else set()
            stats['n_weakly_connected'] = len(wcc)
            stats['largest_wcc_size'] = len(largest_wcc)
            stats['largest_wcc_ratio'] = len(largest_wcc) / n_nodes
        elif n_nodes > 0:
            # Undirected graph fallback
            cc = list(nx.connected_components(G_simple_undir))
            largest_cc = max(cc, key=len) if cc else set()
            stats['n_components'] = len(cc)
            stats['largest_cc_size'] = len(largest_cc)
            stats['largest_cc_ratio'] = len(largest_cc) / n_nodes
        else:
            stats['largest_scc_size'] = 0
            stats['largest_scc_ratio'] = 0
            stats['largest_wcc_size'] = 0
            stats['largest_wcc_ratio'] = 0

        # Clustering coefficient and assortativity (based on simple directed graph,
        # consistent with directed bond percolation)
        G_for_metrics = G_simple_dir if G_simple_dir else G_simple_undir
        if G_for_metrics.number_of_nodes() > 2:
            try:
                stats['clustering_coefficient'] = nx.average_clustering(G_for_metrics)
            except Exception:
                stats['clustering_coefficient'] = 0

            try:
                stats['assortativity'] = nx.degree_assortativity_coefficient(G_for_metrics)
                if np.isnan(stats['assortativity']):
                    stats['assortativity'] = 0
            except Exception:
                stats['assortativity'] = 0
        else:
            stats['clustering_coefficient'] = 0
            stats['assortativity'] = 0

        return stats
