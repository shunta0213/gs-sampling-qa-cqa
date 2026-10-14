import itertools

import networkx as nx
import numpy as np
import pytest

from src.gbp import (
    InstanceSetConfig,
    canonical_partition,
    ensure_instance_set,
    generate_instance_set,
    load_instance,
    load_instance_set,
    solve_problem_from_weights,
)

CONFIG = InstanceSetConfig(n_vertices=8, edge_prob=0.5, weight_range=None, seed=0, min_degenerate=4)


class TestCanonicalPartition:
    def test_first_vertex_is_group_zero(self):
        assert canonical_partition([1, 0, 1, 0]) == (0, 1, 0, 1)

    def test_already_canonical_is_unchanged(self):
        assert canonical_partition([0, 1, 1, 0]) == (0, 1, 1, 0)


class TestGenerateInstanceSet:
    def test_instances_satisfy_all_conditions(self):
        weights, _ = generate_instance_set(CONFIG, n_instances=10)
        assert len(weights) == 10
        for W in weights:
            _, _, assignments = solve_problem_from_weights(W, CONFIG.n_vertices, penalty_offset=1e-3)
            assert len(assignments) >= CONFIG.min_degenerate

    def test_instances_are_pairwise_non_isomorphic(self):
        weights, _ = generate_instance_set(CONFIG, n_instances=10)
        graphs = [nx.from_numpy_array(W) for W in weights]
        assert not any(nx.is_isomorphic(a, b) for a, b in itertools.combinations(graphs, 2))

    def test_prefix_is_stable_when_extending(self):
        short, _ = generate_instance_set(CONFIG, n_instances=3)
        long, _ = generate_instance_set(CONFIG, n_instances=6)
        np.testing.assert_array_equal(short, long[:3])

    def test_returns_fewer_when_draws_run_out(self):
        weights, draw_indices = generate_instance_set(CONFIG, n_instances=100, max_draws=5)
        assert len(weights) < 100
        assert len(weights) == len(draw_indices)


class TestEnsureInstanceSet:
    def test_creates_and_extends_without_changing_existing(self, tmp_path):
        path, n_found = ensure_instance_set(CONFIG, 3, root=tmp_path)
        assert n_found == 3
        _, first = load_instance_set(path)

        _, n_found = ensure_instance_set(CONFIG, 5, root=tmp_path)
        assert n_found == 5
        _, extended = load_instance_set(path)
        np.testing.assert_array_equal(first, extended[:3])

    def test_does_not_rewrite_when_enough(self, tmp_path):
        path, _ = ensure_instance_set(CONFIG, 5, root=tmp_path)
        mtime = path.stat().st_mtime_ns
        _, n_found = ensure_instance_set(CONFIG, 3, root=tmp_path)
        assert n_found == 5
        assert path.stat().st_mtime_ns == mtime

    def test_refuses_to_overwrite_mismatching_file(self, tmp_path):
        path, _ = ensure_instance_set(CONFIG, 3, root=tmp_path)
        with np.load(path) as data:
            tampered = dict(data)
        tampered["W"] = tampered["W"][::-1].copy()
        np.savez(path, **tampered)
        with pytest.raises(RuntimeError):
            ensure_instance_set(CONFIG, 5, root=tmp_path)


class TestLoadInstance:
    def test_loads_config_and_weights(self, tmp_path):
        path, _ = ensure_instance_set(CONFIG, 3, root=tmp_path)
        config, W = load_instance(path, 2)
        assert config == CONFIG
        assert W.shape == (CONFIG.n_vertices, CONFIG.n_vertices)

    def test_out_of_range_index_raises(self, tmp_path):
        path, _ = ensure_instance_set(CONFIG, 3, root=tmp_path)
        with pytest.raises(IndexError):
            load_instance(path, 3)
