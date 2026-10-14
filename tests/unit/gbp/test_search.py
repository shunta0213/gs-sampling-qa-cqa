import numpy as np

from src.gbp import search_problem, solve_problem_from_weights
from src.maxkcut import make_random_edges, make_weight_matrix

# For N=6, independent per-seed streams easily yield the same W (e.g. seeds 2, 3, 33)
ARGS = dict(n_vertices=6, edge_prob=0.8, weight_range=None, penalty_offset=1e-3, min_degenerate=4, max_tries=1000)


def _weights(seed: int) -> np.ndarray:
    return search_problem(seed=seed, build_qutip=False, **ARGS)[2]


class TestSearchProblem:
    def test_weights_are_distinct_across_seeds(self):
        weights = [_weights(seed).tobytes() for seed in range(100)]
        assert len(set(weights)) == 100

    def test_satisfies_min_degenerate(self):
        for seed in range(10):
            assignments = search_problem(seed=seed, build_qutip=False, **ARGS)[7]
            assert len(assignments) >= ARGS["min_degenerate"]

    def test_non_colliding_seed_keeps_first_acceptable_graph(self):
        # seed 0 has no smaller seed to compare against, so it gets the same graph as before deduplication
        rng = np.random.default_rng(0)
        while True:
            W = make_weight_matrix(make_random_edges(6, 0.8, None, rng), 6)
            if len(solve_problem_from_weights(W, 6, 1e-3)[2]) >= 4:
                break
        assert np.array_equal(_weights(0), W)
