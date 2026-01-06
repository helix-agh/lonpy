import igraph as ig
import pandas as pd
import pytest

from lonpy.lon import CMLON, LON, MLON, _contract_vertices, _simplify_with_edge_sum

COLUMNS = ["run", "fit1", "node1", "fit2", "node2"]


@pytest.fixture
def simple_trace_df() -> pd.DataFrame:
    """Simple trace with 3 nodes forming a chain: A -> B -> C (sink)."""
    return pd.DataFrame(
        [
            [0, 100, "A", 50, "B"],
            [0, 50, "B", 10, "C"],
        ],
        columns=COLUMNS,
    )


@pytest.fixture
def multi_run_trace_df() -> pd.DataFrame:
    """Trace with multiple runs visiting same nodes."""
    return pd.DataFrame(
        [
            [0, 100, "A", 50, "B"],
            [0, 50, "B", 10, "C"],
            [1, 100, "A", 50, "B"],
            [1, 50, "B", 10, "C"],
        ],
        columns=COLUMNS,
    )


@pytest.fixture
def neutral_trace_df() -> pd.DataFrame:
    """Trace with neutral (equal-fitness) connections."""
    return pd.DataFrame(
        [
            [0, 100, "A", 50, "B"],
            [0, 50, "B", 50, "C"],  # Equal fitness
            [0, 50, "C", 10, "D"],
        ],
        columns=COLUMNS,
    )


@pytest.fixture
def multiple_sinks_trace_df() -> pd.DataFrame:
    """Trace with multiple sinks (funnels)."""
    return pd.DataFrame(
        [
            [0, 100, "A", 50, "B"],
            [0, 50, "B", 10, "C"],  # C is global sink
            [1, 100, "A", 60, "D"],
            [1, 60, "D", 30, "E"],  # E is local sink
        ],
        columns=COLUMNS,
    )


@pytest.fixture
def worsening_edge_trace_df() -> pd.DataFrame:
    """Trace with worsening edges (escaping)."""
    return pd.DataFrame(
        [
            [0, 100, "A", 50, "B"],
            [0, 50, "B", 80, "C"],  # Worsening edge
            [0, 80, "C", 10, "D"],
        ],
        columns=["run", "fit1", "node1", "fit2", "node2"],
    )


@pytest.fixture
def simple_lon(simple_trace_df) -> LON:
    return LON.from_trace_data(simple_trace_df)


@pytest.fixture
def neutral_lon(neutral_trace_df) -> LON:
    return LON.from_trace_data(neutral_trace_df)


@pytest.fixture
def worsening_lon(worsening_edge_trace_df) -> LON:
    return LON.from_trace_data(worsening_edge_trace_df)


class TestLONFromTraceData:
    def test_creates_lon_from_simple_trace(self, simple_trace_df):
        lon = LON.from_trace_data(simple_trace_df)

        assert isinstance(lon, LON)
        assert isinstance(lon.graph, ig.Graph)

    def test_creates_correct_number_of_vertices(self, simple_trace_df):
        lon = LON.from_trace_data(simple_trace_df)

        assert lon.n_vertices == 3  # A, B, C

    def test_creates_correct_number_of_edges(self, simple_trace_df):
        lon = LON.from_trace_data(simple_trace_df)

        assert lon.n_edges == 2  # A->B, B->C

    def test_sets_best_fitness(self, simple_trace_df):
        lon = LON.from_trace_data(simple_trace_df)

        assert lon.best_fitness == 10  # C has fitness 10

    def test_removes_self_loops(self):
        trace = pd.DataFrame(
            [
                [0, 50, "A", 50, "A"],  # Self-loop
                [0, 50, "A", 30, "B"],
            ],
            columns=COLUMNS,
        )

        lon = LON.from_trace_data(trace)

        assert lon.n_edges == 1


class TestLONProperties:
    def test_n_vertices(self, simple_lon):
        assert simple_lon.n_vertices == 3

    def test_n_edges(self, simple_lon):
        assert simple_lon.n_edges == 2

    def test_vertex_names(self, simple_lon):
        names = simple_lon.vertex_names

        assert isinstance(names, list)
        assert set(names) == {"A", "B", "C"}

    def test_vertex_fitness(self, simple_lon):
        fitness = simple_lon.vertex_fitness

        assert isinstance(fitness, list)
        assert len(fitness) == 3
        assert set(fitness) == {100, 50, 10}

    def test_vertex_count(self, simple_lon):
        counts = simple_lon.vertex_count

        assert isinstance(counts, list)
        assert len(counts) == 3
        assert all(c > 0 for c in counts)


class TestLONGetSinks:
    def test_single_sink(self, simple_lon):
        sinks = simple_lon.get_sinks()

        assert len(sinks) == 1

    def test_multiple_sinks(self, multiple_sinks_trace_df):
        lon = LON.from_trace_data(multiple_sinks_trace_df)

        sinks = lon.get_sinks()

        assert len(sinks) == 2  # C and E are sinks

    def test_sink_has_zero_out_degree(self, simple_lon):
        sinks = simple_lon.get_sinks()

        for sink_idx in sinks:
            out_degree = simple_lon.graph.degree(sink_idx, mode="out")
            assert out_degree == 0


class TestLONGetGlobalOptimaIndices:
    def test_single_global_optimum(self, simple_lon):
        global_indices = simple_lon.get_global_optima_indices()

        assert len(global_indices) == 1
        # Verify it's the node with best fitness
        fitness = simple_lon.vertex_fitness[global_indices[0]]
        assert fitness == simple_lon.best_fitness

    def test_multiple_global_optima(self):
        trace = pd.DataFrame(
            [
                [0, 100, "A", 10, "B"],
                [1, 100, "A", 10, "C"],  # Both B and C have best fitness
            ],
            columns=["run", "fit1", "node1", "fit2", "node2"],
        )

        lon = LON.from_trace_data(trace)
        global_indices = lon.get_global_optima_indices()

        assert len(global_indices) == 2


class TestLONComputeMetrics:
    def test_returns_dict_with_required_keys(self, simple_lon):
        metrics = simple_lon.compute_metrics()

        assert "n_optima" in metrics
        assert "n_funnels" in metrics
        assert "n_global_funnels" in metrics
        assert "neutral" in metrics
        assert "strength" in metrics

    def test_n_optima_metric(self, simple_lon):
        metrics = simple_lon.compute_metrics()

        assert metrics["n_optima"] == 3

    def test_n_funnels_metric(self, simple_lon):
        metrics = simple_lon.compute_metrics()

        assert metrics["n_funnels"] == 1

    def test_n_global_funnels_metric(self, multiple_sinks_trace_df):
        lon = LON.from_trace_data(multiple_sinks_trace_df)
        metrics = lon.compute_metrics()

        assert metrics["n_global_funnels"] == 1  # Only C is at global best

    def test_neutral_metric_with_neutral_edges(self, neutral_lon):
        metrics = neutral_lon.compute_metrics()

        assert metrics["neutral"] > 0  # B and C have neutral connection

    def test_neutral_metric_without_neutral_edges(self, simple_lon):
        metrics = simple_lon.compute_metrics()

        assert metrics["neutral"] == 0.0

    def test_strength_metric(self, simple_lon):
        metrics = simple_lon.compute_metrics()

        # All paths lead to global optimum, so strength should be positive
        assert metrics["strength"] >= 0.0
        assert metrics["strength"] <= 1.0

    def test_known_best_parameter(self, simple_lon):
        # Use a different known_best than what's in the network
        metrics = simple_lon.compute_metrics(known_best=5)

        # No node has fitness 5, so n_global_funnels should be 0
        assert metrics["n_global_funnels"] == 0


class TestLONClassifyEdges:
    def test_classifies_improving_edges(self, simple_lon):
        simple_lon.classify_edges()

        edge_types = simple_lon.graph.es["edge_type"]
        assert "improving" in edge_types

    def test_classifies_equal_edges(self, neutral_lon):
        neutral_lon.classify_edges()

        edge_types = neutral_lon.graph.es["edge_type"]
        assert "equal" in edge_types

    def test_classifies_worsening_edges(self, worsening_lon):
        worsening_lon.classify_edges()

        edge_types = worsening_lon.graph.es["edge_type"]
        assert "worsening" in edge_types

    def test_handles_empty_graph(self):
        lon = LON()
        lon.classify_edges()  # Should not raise

    def test_adds_edge_type_attribute(self, simple_lon):
        simple_lon.classify_edges()

        assert "edge_type" in simple_lon.graph.es.attributes()


class TestLONToMLON:
    def test_returns_mlon(self, simple_lon):
        mlon = simple_lon.to_mlon()

        assert isinstance(mlon, MLON)

    def test_mlon_has_reference_to_source_lon(self, simple_lon):
        mlon = simple_lon.to_mlon()

        assert mlon.source_lon is simple_lon


class TestLONToCMLON:
    def test_returns_cmlon(self, simple_lon):
        cmlon = simple_lon.to_cmlon()

        assert isinstance(cmlon, CMLON)

    def test_cmlon_has_reference_to_source_lon(self, simple_lon):
        cmlon = simple_lon.to_cmlon()

        assert cmlon.source_lon is simple_lon


class TestMLONFromLON:
    def test_creates_mlon_from_lon(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        assert isinstance(mlon, MLON)

    def test_preserves_all_vertices(self, worsening_lon):
        mlon = MLON.from_lon(worsening_lon)

        assert mlon.n_vertices == worsening_lon.n_vertices

    def test_removes_worsening_edges(self, worsening_lon):
        mlon = MLON.from_lon(worsening_lon)

        # Original has 3 edges, one is worsening
        assert mlon.n_edges < worsening_lon.n_edges

    def test_keeps_improving_edges(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        # All edges in simple_lon are improving
        assert mlon.n_edges == simple_lon.n_edges

    def test_keeps_equal_edges(self, neutral_lon):
        mlon = MLON.from_lon(neutral_lon)

        assert mlon.n_edges == neutral_lon.n_edges

    def test_preserves_best_fitness(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        assert mlon.best_fitness == simple_lon.best_fitness

    def test_handles_empty_lon(self):
        lon = LON()
        mlon = MLON.from_lon(lon)

        assert mlon.n_vertices == 0
        assert mlon.n_edges == 0


class TestMLONProperties:
    def test_n_vertices(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        assert mlon.n_vertices == 3

    def test_n_edges(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        assert mlon.n_edges == 2

    def test_vertex_fitness(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        assert isinstance(mlon.vertex_fitness, list)
        assert len(mlon.vertex_fitness) == 3


class TestMLONGetSinks:
    def test_identifies_sinks(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        sinks = mlon.get_sinks()

        assert len(sinks) >= 1

    def test_worsening_edges_create_more_sinks(self, worsening_lon):
        mlon = MLON.from_lon(worsening_lon)

        mlon_sinks = mlon.get_sinks()
        lon_sinks = worsening_lon.get_sinks()

        # After removing worsening B->C, B becomes a sink
        assert len(mlon_sinks) >= len(lon_sinks)


class TestMLONGetGlobalOptimaIndices:
    def test_identifies_global_optima(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)

        global_indices = mlon.get_global_optima_indices()

        assert len(global_indices) >= 1


class TestMLONToCMLON:
    def test_returns_cmlon(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)
        cmlon = mlon.to_cmlon()

        assert isinstance(cmlon, CMLON)

    def test_uses_source_lon_when_available(self, simple_lon):
        mlon = MLON.from_lon(simple_lon)
        cmlon = mlon.to_cmlon()

        assert cmlon.source_lon is simple_lon


class TestCMLONFromLON:
    def test_creates_cmlon_from_lon(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert isinstance(cmlon, CMLON)

    def test_contracts_neutral_nodes(self, neutral_lon):
        cmlon = CMLON.from_lon(neutral_lon)

        # B and C have equal fitness and are connected, should be contracted
        assert cmlon.n_vertices < neutral_lon.n_vertices

    def test_no_contraction_without_neutral_edges(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert cmlon.n_vertices == simple_lon.n_vertices

    def test_preserves_best_fitness(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert cmlon.best_fitness == simple_lon.best_fitness

    def test_sums_edge_counts(self, multi_run_trace_df):
        lon = LON.from_trace_data(multi_run_trace_df)
        cmlon = CMLON.from_lon(lon)

        if cmlon.n_edges > 0 and "Count" in cmlon.graph.es.attributes():
            edge_counts = cmlon.graph.es["Count"]
            assert all(c >= 1 for c in edge_counts)

    def test_handles_empty_lon(self):
        lon = LON()
        cmlon = CMLON.from_lon(lon)

        assert cmlon.n_vertices == 0
        assert cmlon.n_edges == 0


class TestCMLONProperties:
    def test_n_vertices(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert cmlon.n_vertices == 3

    def test_n_edges(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert cmlon.n_edges == 2

    def test_vertex_fitness(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        assert isinstance(cmlon.vertex_fitness, list)
        assert len(cmlon.vertex_fitness) == cmlon.n_vertices

    def test_vertex_count(self, neutral_lon):
        cmlon = CMLON.from_lon(neutral_lon)

        # Some contracted nodes should have count > 1
        assert isinstance(cmlon.vertex_count, list)
        assert len(cmlon.vertex_count) == cmlon.n_vertices


class TestCMLONGetSinks:
    def test_identifies_sinks(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        sinks = cmlon.get_sinks()

        assert len(sinks) >= 1

    def test_sink_has_zero_out_degree(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        sinks = cmlon.get_sinks()
        for sink_idx in sinks:
            out_degree = cmlon.graph.degree(sink_idx, mode="out")
            assert out_degree == 0


class TestCMLONGetGlobalSinks:
    def test_identifies_global_sinks(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        global_sinks = cmlon.get_global_sinks()

        assert len(global_sinks) >= 1
        for sink_idx in global_sinks:
            assert cmlon.vertex_fitness[sink_idx] == cmlon.best_fitness


class TestCMLONGetLocalSinks:
    def test_identifies_local_sinks(self, multiple_sinks_trace_df):
        lon = LON.from_trace_data(multiple_sinks_trace_df)
        cmlon = CMLON.from_lon(lon)

        local_sinks = cmlon.get_local_sinks()

        for sink_idx in local_sinks:
            assert cmlon.vertex_fitness[sink_idx] > cmlon.best_fitness

    def test_no_local_sinks_when_all_global(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        local_sinks = cmlon.get_local_sinks()

        # Simple LON has only one sink which is global
        assert len(local_sinks) == 0


class TestCMLONComputeMetrics:
    def test_returns_dict_with_required_keys(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)
        metrics = cmlon.compute_metrics()

        assert "n_optima" in metrics
        assert "n_funnels" in metrics
        assert "n_global_funnels" in metrics
        assert "neutral" in metrics
        assert "strength" in metrics
        assert "global_funnel_proportion" in metrics

    def test_n_optima_metric(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)
        metrics = cmlon.compute_metrics()

        assert metrics["n_optima"] == cmlon.n_vertices

    def test_neutral_metric_with_contraction(self, neutral_lon):
        cmlon = CMLON.from_lon(neutral_lon)
        metrics = cmlon.compute_metrics()

        # neutral = 1 - cmlon.n_vertices / lon.n_vertices
        expected_neutral = 1.0 - cmlon.n_vertices / neutral_lon.n_vertices
        assert metrics["neutral"] == round(expected_neutral, 4)

    def test_strength_metric(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)
        metrics = cmlon.compute_metrics()

        assert metrics["strength"] >= 0.0
        assert metrics["strength"] <= 1.0

    def test_global_funnel_proportion_metric(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)
        metrics = cmlon.compute_metrics()

        assert metrics["global_funnel_proportion"] >= 0.0
        assert metrics["global_funnel_proportion"] <= 1.0


class TestCMLONGlobalFunnelProportion:
    def test_all_nodes_reach_global(self, simple_lon):
        cmlon = CMLON.from_lon(simple_lon)

        proportion = cmlon._compute_global_funnel_proportion()

        # All nodes in chain lead to global optimum
        assert proportion == 1.0

    def test_partial_reach(self, multiple_sinks_trace_df):
        lon = LON.from_trace_data(multiple_sinks_trace_df)
        cmlon = CMLON.from_lon(lon)

        proportion = cmlon._compute_global_funnel_proportion()

        # D and E lead to local sink, not all nodes reach global
        assert proportion < 1.0
        assert proportion > 0.0


class TestContractVertices:
    def test_contracts_by_membership(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(4)
        graph.vs["name"] = ["A", "B", "C", "D"]
        graph.vs["Fitness"] = [10, 10, 20, 30]
        graph.vs["Count"] = [1, 1, 1, 1]
        graph.add_edges([(0, 1), (1, 2), (2, 3)])
        graph.es["Count"] = [1, 1, 1]

        # Contract A and B (membership 0), C and D separate
        membership = [0, 0, 1, 2]

        result = _contract_vertices(
            graph,
            membership,
            vertex_attr_comb={"name": "first", "Fitness": "first", "Count": "sum"},
        )

        assert result.vcount() == 3

    def test_sums_count_attribute(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = ["A", "B"]
        graph.vs["Fitness"] = [10, 10]
        graph.vs["Count"] = [3, 5]
        graph.add_edges([(0, 1)])
        graph.es["Count"] = [1]

        membership = [0, 0]  # Contract both

        result = _contract_vertices(
            graph,
            membership,
            vertex_attr_comb={"name": "first", "Fitness": "first", "Count": "sum"},
        )

        assert result.vs[0]["Count"] == 8

    def test_removes_internal_edges(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(3)
        graph.vs["name"] = ["A", "B", "C"]
        graph.vs["Fitness"] = [10, 10, 20]
        graph.add_edges([(0, 1), (1, 2)])  # A->B (internal), B->C (external)
        graph.es["Count"] = [1, 1]

        membership = [0, 0, 1]

        result = _contract_vertices(
            graph,
            membership,
            vertex_attr_comb={"name": "first", "Fitness": "first"},
        )

        # Only B->C should remain (as contracted_0 -> 1)
        assert result.ecount() == 1


class TestSimplifyWithEdgeSum:
    def test_removes_self_loops(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = ["A", "B"]
        graph.vs["Fitness"] = [10, 20]
        graph.add_edges([(0, 0), (0, 1)])  # Self-loop and normal edge
        graph.es["Count"] = [1, 1]

        result = _simplify_with_edge_sum(graph)

        assert result.ecount() == 1

    def test_combines_parallel_edges(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = ["A", "B"]
        graph.vs["Fitness"] = [10, 20]
        graph.add_edges([(0, 1), (0, 1)])  # Parallel edges
        graph.es["Count"] = [3, 5]

        result = _simplify_with_edge_sum(graph)

        assert result.ecount() == 1
        assert result.es[0]["Count"] == 8

    def test_preserves_vertex_attributes(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = ["A", "B"]
        graph.vs["Fitness"] = [10, 20]
        graph.vs["Count"] = [1, 2]
        graph.add_edges([(0, 1)])
        graph.es["Count"] = [1]

        result = _simplify_with_edge_sum(graph)

        assert result.vs["name"] == ["A", "B"]
        assert result.vs["Fitness"] == [10, 20]
        assert result.vs["Count"] == [1, 2]

    def test_handles_empty_graph(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = ["A", "B"]
        graph.vs["Fitness"] = [10, 20]

        result = _simplify_with_edge_sum(graph)

        assert result.vcount() == 2
        assert result.ecount() == 0


class TestLONIntegration:
    def test_lon_to_mlon_to_cmlon(self, neutral_lon):
        mlon = neutral_lon.to_mlon()
        cmlon = mlon.to_cmlon()

        assert isinstance(cmlon, CMLON)
        assert cmlon.source_lon is neutral_lon

    def test_lon_direct_to_cmlon(self, neutral_lon):
        cmlon = neutral_lon.to_cmlon()

        assert isinstance(cmlon, CMLON)
        assert cmlon.n_vertices <= neutral_lon.n_vertices

    def test_metrics_consistency(self, simple_lon):
        lon_metrics = simple_lon.compute_metrics()
        cmlon = simple_lon.to_cmlon()
        cmlon_metrics = cmlon.compute_metrics()

        # For simple LON without neutral edges, metrics should match
        assert lon_metrics["n_optima"] == cmlon_metrics["n_optima"]
        assert lon_metrics["n_funnels"] == cmlon_metrics["n_funnels"]
        assert lon_metrics["n_global_funnels"] == cmlon_metrics["n_global_funnels"]

    def test_complex_landscape(self):
        trace = pd.DataFrame(
            [
                # Run 0: A -> B -> C (global)
                [0, 100, "A", 50, "B"],
                [0, 50, "B", 10, "C"],
                # Run 1: A -> D -> E (local sink)
                [1, 100, "A", 70, "D"],
                [1, 70, "D", 40, "E"],
                # Run 2: F -> G -> H (neutral) -> C (global)
                [2, 90, "F", 60, "G"],
                [2, 60, "G", 60, "H"],
                [2, 60, "H", 10, "C"],
            ],
            columns=["run", "fit1", "node1", "fit2", "node2"],
        )

        lon = LON.from_trace_data(trace)
        metrics = lon.compute_metrics()

        assert metrics["n_optima"] == 8
        assert metrics["n_funnels"] == 2  # C and E
        assert metrics["n_global_funnels"] == 1  # Only C
        assert metrics["neutral"] > 0  # G and H are neutral

        cmlon = lon.to_cmlon()
        cmlon_metrics = cmlon.compute_metrics()

        # CMLON should have fewer vertices due to G-H contraction
        assert cmlon.n_vertices < lon.n_vertices
        assert cmlon_metrics["neutral"] > 0
