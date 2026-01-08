import contextlib
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import pandas as pd


@dataclass
class LON:
    """
    Local Optima Network (LON) representation.

    A LON is a directed graph where nodes represent local optima and edges
    represent transitions between them discovered during basin-hopping search.

    Attributes:
        graph: The underlying igraph Graph object.
        best_fitness: The best (minimum) fitness value found.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None

    @classmethod
    def from_trace_data(
        cls,
        trace: pd.DataFrame,
    ) -> "LON":
        """
        Create a LON from trace data.

        Args:
            trace: DataFrame with columns [run, fit1, node1, fit2, node2] where:
                - run: integer run number
                - fit1: integer fitness of source node (scaled)
                - node1: string hash of source node
                - fit2: integer fitness of target node (scaled)
                - node2: string hash of target node

        Returns:
            LON instance with constructed graph.
        """
        trace = trace.copy()
        trace.columns = pd.Index(["run", "fit1", "node1", "fit2", "node2"])

        # Combine nodes from both columns
        lnodes1 = trace[["node1", "fit1"]].rename(columns={"node1": "Node", "fit1": "Fitness"})
        lnodes2 = trace[["node2", "fit2"]].rename(columns={"node2": "Node", "fit2": "Fitness"})
        lnodes = pd.concat([lnodes1, lnodes2], ignore_index=True)

        ledges = trace[["node1", "node2"]].copy()

        # Count node and edge occurrences
        nodes = lnodes.groupby(["Node", "Fitness"], as_index=False).size()
        nodes.columns = pd.Index(["Node", "Fitness", "Count"])

        edges = ledges.groupby(["node1", "node2"], as_index=False).size()
        edges.columns = pd.Index(["Start", "End", "Count"])

        graph = ig.Graph(directed=True)

        for _, row in nodes.iterrows():
            graph.add_vertex(name=str(row["Node"]), Fitness=row["Fitness"], Count=row["Count"])

        for _, row in edges.iterrows():
            with contextlib.suppress(ValueError):
                graph.add_edge(str(row["Start"]), str(row["End"]), Count=row["Count"])

        # Remove self-loops
        graph = graph.simplify(multiple=False, loops=True)

        best = nodes["Fitness"].min()

        return cls(graph=graph, best_fitness=best)

    @property
    def n_vertices(self) -> int:
        """Number of vertices (local optima) in the LON."""
        return int(self.graph.vcount())

    @property
    def n_edges(self) -> int:
        """Number of edges in the LON."""
        return int(self.graph.ecount())

    @property
    def vertex_names(self) -> list[str]:
        """List of vertex names (node hashes)."""
        return list(self.graph.vs["name"])

    @property
    def vertex_fitness(self) -> list[float]:
        """List of vertex fitness values."""
        return list(self.graph.vs["Fitness"])

    @property
    def vertex_count(self) -> list[int]:
        """List of vertex counts (times visited)."""
        return list(self.graph.vs["Count"])

    def get_sinks(self) -> list[int]:
        """Get indices of sink nodes (nodes with no outgoing edges)."""
        out_degrees = self.graph.degree(mode="out")
        return [i for i, d in enumerate(out_degrees) if d == 0]

    def get_global_optima_indices(self) -> list[int]:
        """Get indices of global optima nodes (nodes at best fitness)."""
        return [i for i, f in enumerate(self.vertex_fitness) if f == self.best_fitness]

    def compute_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute LON metrics.

        Args:
            known_best: Known global optimum value. If None, uses the best
                fitness found in the network.

        Returns:
            Dictionary containing:
                - n_optima: Number of local optima (vertices)
                - n_funnels: Number of funnels (sinks)
                - n_global_funnels: Number of funnels at global optimum
                - neutral: Proportion of nodes with equal-fitness connections
                - strength: Proportion of incoming strength to global optima
        """
        best = known_best if known_best is not None else self.best_fitness

        n_optima = self.n_vertices

        sinks_id = self.get_sinks()
        n_funnels = len(sinks_id)

        sinks_fit = [self.vertex_fitness[i] for i in sinks_id]
        n_global_funnels = sum(1 for f in sinks_fit if f == best)

        # Neutral: proportion of nodes with equal-fitness connections
        el = self.graph.get_edgelist()
        fits = self.vertex_fitness
        neutral_edge_indices = []
        for i, (src, tgt) in enumerate(el):
            if fits[src] == fits[tgt]:
                neutral_edge_indices.append(i)

        if neutral_edge_indices:
            gnn = self.graph.subgraph_edges(neutral_edge_indices, delete_vertices=True)
            gnn = gnn.simplify(multiple=False, loops=True)
            neutral = round(gnn.vcount() / n_optima, 4)
        else:
            neutral = 0.0

        # Strength: incoming strength to global optima
        igs = self.get_global_optima_indices()
        if self.n_edges > 0 and igs:
            edge_weights = self.graph.es["Count"]
            stren_igs = sum(self.graph.strength(igs, mode="in", loops=False, weights=edge_weights))
            stren_all = sum(self.graph.strength(mode="in", loops=False, weights=edge_weights))
            strength = round(stren_igs / stren_all, 4) if stren_all > 0 else 0.0
        else:
            strength = 0.0

        return {
            "n_optima": n_optima,
            "n_funnels": n_funnels,
            "n_global_funnels": n_global_funnels,
            "neutral": neutral,
            "strength": strength,
        }

    def classify_edges(self) -> None:
        """
        Classify edges as 'improving', 'equal', or 'worsening'.

        Adds 'edge_type' attribute to each edge based on fitness comparison:
        - 'improving': target has better (lower) fitness than source
        - 'equal': source and target have same fitness
        - 'worsening': target has worse (higher) fitness than source

        This classification is used for MLON construction and visualization.
        """
        if self.n_edges == 0:
            return

        fits = self.vertex_fitness
        edge_types = []

        for edge in self.graph.es:
            src_fit = fits[edge.source]
            tgt_fit = fits[edge.target]

            if tgt_fit < src_fit:
                edge_types.append("improving")
            elif tgt_fit > src_fit:
                edge_types.append("worsening")
            else:
                edge_types.append("equal")

        self.graph.es["edge_type"] = edge_types

    def to_mlon(self) -> "MLON":
        """
        Convert LON to Monotonic LON (MLON).

        MLON keeps only non-worsening edges (improving and equal).
        All vertices are preserved even if they become disconnected.

        Returns:
            MLON instance with only monotonic edges.
        """
        return MLON.from_lon(self)

    def to_cmlon(self) -> "CMLON":
        """
        Convert LON to Compressed Monotonic LON (CMLON).

        Returns:
            CMLON instance with contracted neutral nodes.
        """
        return CMLON.from_lon(self)


@dataclass
class MLON:
    """
    Monotonic Local Optima Network (MLON).

    MLON is a variant of LON that keeps only non-worsening edges
    (improving and equal fitness transitions). This focuses on
    the "downhill" structure of the landscape.

    Attributes:
        graph: The underlying igraph Graph object.
        best_fitness: The best (minimum) fitness value.
        source_lon: Reference to the original LON.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    source_lon: LON | None = None

    @classmethod
    def from_lon(cls, lon: LON) -> "MLON":
        """
        Create MLON from LON by removing worsening edges.

        Args:
            lon: Source LON instance.

        Returns:
            MLON with only improving and equal edges.
        """
        if lon.n_edges == 0:
            mlon_graph = lon.graph.copy()
            return cls(graph=mlon_graph, best_fitness=lon.best_fitness, source_lon=lon)

        # Ensure edges are classified
        if "edge_type" not in lon.graph.es.attributes():
            lon.classify_edges()

        # Keep only non-worsening edges
        non_worsening_indices = [i for i, e in enumerate(lon.graph.es) if e["edge_type"] != "worsening"]

        # Create subgraph keeping all vertices
        mlon_graph = lon.graph.subgraph_edges(non_worsening_indices, delete_vertices=False)
        mlon_graph = mlon_graph.simplify(multiple=False, loops=True)

        return cls(graph=mlon_graph, best_fitness=lon.best_fitness, source_lon=lon)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in MLON."""
        return int(self.graph.vcount())

    @property
    def n_edges(self) -> int:
        """Number of edges in MLON."""
        return int(self.graph.ecount())

    @property
    def vertex_fitness(self) -> list[float]:
        """List of vertex fitness values."""
        return list(self.graph.vs["Fitness"])

    def get_sinks(self) -> list[int]:
        """Get indices of sink nodes (nodes with no outgoing edges)."""
        out_degrees = self.graph.degree(mode="out")
        return [i for i, d in enumerate(out_degrees) if d == 0]

    def get_global_optima_indices(self) -> list[int]:
        """Get indices of global optima nodes."""
        return [i for i, f in enumerate(self.vertex_fitness) if f == self.best_fitness]

    def to_cmlon(self) -> "CMLON":
        """
        Convert MLON to Compressed Monotonic LON (CMLON).

        Returns:
            CMLON instance with contracted neutral nodes.
        """
        # Use the source LON if available for proper CMLON construction
        if self.source_lon is not None:
            return CMLON.from_lon(self.source_lon)

        # Otherwise create a temporary LON-like structure
        temp_lon = LON(graph=self.graph.copy(), best_fitness=self.best_fitness)
        return CMLON.from_lon(temp_lon)


@dataclass
class CMLON:
    """
    Compressed Monotonic Local Optima Network (CMLON).

    CMLON contracts nodes with equal fitness that are connected,
    creating a compressed representation of the fitness landscape.

    Attributes:
        graph: The underlying igraph Graph object.
        best_fitness: The best (minimum) fitness value.
        source_lon: Reference to the original LON (optional).
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    source_lon: LON | None = None

    @classmethod
    def from_lon(cls, lon: LON) -> "CMLON":
        """
        Create CMLON from LON by contracting neutral nodes.

        The compression process:
        1. Mark edges as "improving" (f2 < f1) or "equal" (f2 == f1)
        2. Create subgraph of equal-fitness edges
        3. Find weakly connected components
        4. Contract vertices using component membership
        5. Combine parallel edge weights

        Args:
            lon: Source LON instance.

        Returns:
            CMLON with contracted neutral components.
        """
        if lon.n_edges == 0:
            cmlon_graph = lon.graph.copy()
            return cls(graph=cmlon_graph, best_fitness=lon.best_fitness, source_lon=lon)

        # Create a working copy
        mlon = lon.graph.copy()
        mlon.vs["Count"] = [1] * mlon.vcount()

        el = mlon.get_edgelist()
        fits = mlon.vs["Fitness"]

        f1 = [fits[src] for src, _ in el]
        f2 = [fits[tgt] for _, tgt in el]

        # Mark edge types and find equal-fitness edges
        edge_types = []
        equal_edge_indices = []
        for i, (fit1, fit2) in enumerate(zip(f1, f2, strict=True)):
            if fit2 < fit1:
                edge_types.append("improving")
            elif fit2 == fit1:
                edge_types.append("equal")
                equal_edge_indices.append(i)
            else:
                edge_types.append("worsening")
        mlon.es["type"] = edge_types

        # Create subgraph of equal-fitness edges (keep all vertices)
        if equal_edge_indices:
            gnn = mlon.subgraph_edges(equal_edge_indices, delete_vertices=False)
        else:
            gnn = ig.Graph(n=mlon.vcount(), directed=True)

        # Find weakly connected components
        nn_memb = gnn.components(mode="weak").membership

        # Contract vertices using component membership
        cmlon_graph = _contract_vertices(
            mlon,
            nn_memb,
            vertex_attr_comb={"Fitness": "first", "Count": "sum", "name": "first"},
        )

        # Combine parallel edges by summing Count
        cmlon_graph = _simplify_with_edge_sum(cmlon_graph)

        return cls(graph=cmlon_graph, best_fitness=lon.best_fitness, source_lon=lon)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in CMLON."""
        return int(self.graph.vcount())

    @property
    def n_edges(self) -> int:
        """Number of edges in CMLON."""
        return int(self.graph.ecount())

    @property
    def vertex_fitness(self) -> list[float]:
        """List of vertex fitness values."""
        return list(self.graph.vs["Fitness"])

    @property
    def vertex_count(self) -> list[int]:
        """List of vertex counts (contracted nodes)."""
        return list(self.graph.vs["Count"])

    def get_sinks(self) -> list[int]:
        """Get indices of sink nodes (nodes with no outgoing edges)."""
        out_degrees = self.graph.degree(mode="out")
        return [i for i, d in enumerate(out_degrees) if d == 0]

    def get_global_sinks(self) -> list[int]:
        """Get indices of global sinks (sinks at best fitness)."""
        sinks = self.get_sinks()
        fits = self.vertex_fitness
        return [s for s in sinks if fits[s] == self.best_fitness]

    def get_local_sinks(self) -> list[int]:
        """Get indices of local sinks (sinks not at best fitness)."""
        sinks = self.get_sinks()
        fits = self.vertex_fitness
        if self.best_fitness is None:
            return []
        return [s for s in sinks if fits[s] > self.best_fitness]

    def compute_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute CMLON metrics.

        Args:
            known_best: Known global optimum value. If None, uses the best
                fitness found in the network.

        Returns:
            Dictionary containing:
                - n_optima: Number of optima in CMLON
                - n_funnels: Number of funnels (sinks)
                - n_global_funnels: Number of funnels at global optimum
                - neutral: Proportion of contracted nodes
                - strength: Ratio of incoming strength to global vs local sinks
                - global_funnel_proportion: Proportion of nodes that can reach
                  a global optimum
        """
        best = known_best if known_best is not None else self.best_fitness

        n_optima = self.n_vertices

        sinks_id = self.get_sinks()
        n_funnels = len(sinks_id)

        sinks_fit = [self.vertex_fitness[i] for i in sinks_id]
        n_global_funnels = sum(1 for f in sinks_fit if f == best)

        # Neutral: proportion of contracted nodes
        neutral = round(1.0 - self.n_vertices / self.source_lon.n_vertices, 4) if self.source_lon is not None else 0.0

        # Strength: ratio of incoming strength to global vs local sinks
        igs = [s for s, f in zip(sinks_id, sinks_fit, strict=True) if f == best]
        ils = [s for s, f in zip(sinks_id, sinks_fit, strict=True) if best is not None and f > best]

        if self.n_edges > 0:
            edge_weights = self.graph.es["Count"] if "Count" in self.graph.es.attributes() else None
            sing = sum(self.graph.strength(igs, mode="in", loops=False, weights=edge_weights)) if igs else 0
            sinl = sum(self.graph.strength(ils, mode="in", loops=False, weights=edge_weights)) if ils else 0
            total = sing + sinl
            strength = round(sing / total, 4) if total > 0 else 0.0
        else:
            strength = 0.0

        gfunnel = self._compute_global_funnel_proportion()

        return {
            "n_optima": n_optima,
            "n_funnels": n_funnels,
            "n_global_funnels": n_global_funnels,
            "neutral": neutral,
            "strength": strength,
            "global_funnel_proportion": gfunnel,
        }

    def _compute_global_funnel_proportion(self) -> float:
        """Compute proportion of nodes that can reach a global optimum."""
        igs = self.get_global_sinks()
        if not igs:
            return 0.0

        # Get all nodes that can reach any global sink
        reachable = set()
        for sink in igs:
            component = self.graph.subcomponent(sink, mode="in")
            reachable.update(component)

        return len(reachable) / self.n_vertices if self.n_vertices > 0 else 0.0


def _contract_vertices(
    graph: ig.Graph,
    membership: list[int],
    vertex_attr_comb: dict[str, str],
) -> ig.Graph:
    """
    Contract vertices according to membership, combining attributes.

    Args:
        graph: Input graph.
        membership: Component membership for each vertex.
        vertex_attr_comb: How to combine vertex attributes. Supported methods:
            "first", "sum", "min", "max", "ignore".

    Returns:
        New graph with contracted vertices.
    """
    n_components = max(membership) + 1

    # Group vertices by component
    components: dict[int, list[int]] = {i: [] for i in range(n_components)}
    for v_idx, comp in enumerate(membership):
        components[comp].append(v_idx)

    # Create new graph
    new_graph = ig.Graph(directed=True)
    new_graph.add_vertices(n_components)

    # Combine vertex attributes
    for attr in graph.vs.attributes():
        comb_method = vertex_attr_comb.get(attr, "ignore")
        if comb_method == "ignore":
            continue

        new_values = []
        for comp_idx in range(n_components):
            verts = components[comp_idx]
            values = [graph.vs[v][attr] for v in verts]

            if comb_method == "first":
                new_values.append(values[0] if values else None)
            elif comb_method == "sum":
                new_values.append(sum(values))
            elif comb_method == "min":
                new_values.append(min(values))
            elif comb_method == "max":
                new_values.append(max(values))
            else:
                new_values.append(values[0] if values else None)

        new_graph.vs[attr] = new_values

    # Map old edges to new edges
    new_edges: dict[tuple[int, int], float] = {}
    for edge in graph.es:
        src_comp = membership[edge.source]
        tgt_comp = membership[edge.target]
        if src_comp != tgt_comp:  # Skip self-loops created by contraction
            key = (src_comp, tgt_comp)
            edge_count = edge["Count"] if "Count" in edge.attributes() else 1
            if key in new_edges:
                new_edges[key] += edge_count
            else:
                new_edges[key] = edge_count

    # Add edges
    if new_edges:
        edges = list(new_edges.keys())
        counts = list(new_edges.values())
        new_graph.add_edges(edges)
        new_graph.es["Count"] = counts

    return new_graph


def _simplify_with_edge_sum(graph: ig.Graph) -> ig.Graph:
    """
    Simplify graph by removing self-loops and combining parallel edges.

    For parallel edges, sums the Count attribute.
    """
    # Remove self-loops
    graph = graph.simplify(multiple=False, loops=True)

    # Combine parallel edges
    if graph.ecount() == 0:
        return graph

    edge_dict: dict[tuple[int, int], float] = {}
    for edge in graph.es:
        key = (edge.source, edge.target)
        count = edge["Count"] if "Count" in edge.attributes() else 1
        if key in edge_dict:
            edge_dict[key] += count
        else:
            edge_dict[key] = count

    # Rebuild graph with combined edges
    new_graph = ig.Graph(directed=True)
    new_graph.add_vertices(graph.vcount())

    # Copy vertex attributes
    for attr in graph.vs.attributes():
        new_graph.vs[attr] = graph.vs[attr]

    if edge_dict:
        edges = list(edge_dict.keys())
        counts = list(edge_dict.values())
        new_graph.add_edges(edges)
        new_graph.es["Count"] = counts

    return new_graph
