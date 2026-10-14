"""Generation, saving and loading of Graph Bipartitioning problem instance sets.

search.search_problem searches graphs from an independent RNG stream per seed, so
isomorphic graphs (differing only by vertex relabeling) can arise from different seeds.

This module draws graphs in order from a single RNG stream and adopts, in order, only
those that

1. have a degeneracy (number of assignments, including label swaps) of at least min_degenerate
2. are not isomorphic to any previously adopted graph

numbering them as instance index 0, 1, 2, .... The adoption decision depends only on the
previously adopted results, so regenerating with the same settings and a larger
n_instances reproduces the leading part exactly (existing instance indices do not change).

Instances whose optimal-solution degeneracy stems from graph symmetry (automorphisms) are not excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from ..maxkcut import exact_min_penalty_from_diag, make_random_edges, make_weight_matrix
from .hamiltonian import gbp_diag_numpy
from .solver import solve_gbp_from_diag

# Bump when the generation rules change (recorded in saved files and checked on load)
GENERATOR_VERSION = 2

INSTANCES_ROOT = Path(__file__).resolve().parents[2] / "data" / "gbp-instances"

# Tolerance for weight equality (used for weighted-graph isomorphism checks)
_WEIGHT_ATOL = 1e-9


@dataclass(frozen=True)
class InstanceSetConfig:
    """Generation parameters that uniquely determine an instance set."""

    n_vertices: int
    edge_prob: float
    weight_range: tuple[float, float] | None
    seed: int
    min_degenerate: int

    def file_name(self) -> str:
        weight = "unweighted" if self.weight_range is None else f"w{self.weight_range[0]:g}-{self.weight_range[1]:g}"
        return (
            f"gbp-v{GENERATOR_VERSION}-n{self.n_vertices}-p{self.edge_prob:g}-{weight}"
            f"-deg{self.min_degenerate}-seed{self.seed}.npz"
        )

    def path(self, root: Path = INSTANCES_ROOT) -> Path:
        return root / self.file_name()


def solve_problem_from_weights(
    W: np.ndarray,
    n_vertices: int,
    penalty_offset: float,
    diag_chunk_size: int | None = None,
) -> tuple[float, float, list[list[int]]]:
    """From the weight matrix, compute the exact penalty lower bound A*, the actual penalty, and the degenerate optimal assignments."""
    diag_c, diag_p = gbp_diag_numpy(W, n_vertices, chunk_size=diag_chunk_size)
    penalty_min = exact_min_penalty_from_diag(diag_c, diag_p)
    penalty = penalty_min + penalty_offset
    assignments = solve_gbp_from_diag(diag_c, diag_p, n_vertices, penalty)
    return penalty_min, penalty, assignments


def canonical_partition(assignment) -> tuple[int, ...]:
    """Normalize the group 0/1 label-swap symmetry so that the first vertex is always 0."""
    assignment = tuple(int(b) for b in assignment)
    if assignment[0] == 1:
        return tuple(1 - b for b in assignment)
    return assignment


def weighted_graph(W: np.ndarray) -> nx.Graph:
    """Build a networkx graph (with weight attributes) from the weight matrix. Isolated vertices are kept."""
    n_vertices = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for i, j in zip(*np.nonzero(np.triu(W, k=1))):
        G.add_edge(int(i), int(j), weight=float(W[i, j]))
    return G


def _edge_match(a: dict, b: dict) -> bool:
    return abs(a["weight"] - b["weight"]) <= _WEIGHT_ATOL


class _IsomorphismRegistry:
    """Bucket adopted graphs by WL hash and prevent registering isomorphic duplicates."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[nx.Graph]] = {}

    @staticmethod
    def _key(G: nx.Graph) -> str:
        H = G.copy()
        nx.set_edge_attributes(H, {e: f"{w:.9g}" for e, w in nx.get_edge_attributes(G, "weight").items()}, "wkey")
        return nx.weisfeiler_lehman_graph_hash(H, edge_attr="wkey")

    def contains(self, G: nx.Graph) -> bool:
        return any(nx.is_isomorphic(G, H, edge_match=_edge_match) for H in self._buckets.get(self._key(G), []))

    def add(self, G: nx.Graph) -> None:
        self._buckets.setdefault(self._key(G), []).append(G)


def generate_instance_set(
    config: InstanceSetConfig,
    n_instances: int,
    max_draws: int = 100_000,
    penalty_offset: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Adopt, in order, up to n_instances graphs satisfying the conditions from config's RNG stream.

    If n_instances is not reached after max_draws draws, return only those adopted so far
    (the caller should check the count).

    Returns
    -------
    weights : np.ndarray, shape (n_found, n_vertices, n_vertices)
    draw_indices : np.ndarray, shape (n_found,)
        The draw number in the RNG stream at which each instance was obtained (for the record)
    """
    if config.n_vertices % 2 != 0:
        raise ValueError("n_vertices must be even for balanced bipartitioning")

    rng = np.random.default_rng(config.seed)
    registry = _IsomorphismRegistry()
    weights: list[np.ndarray] = []
    draw_indices: list[int] = []
    for draw in range(max_draws):
        if len(weights) >= n_instances:
            break
        edges = make_random_edges(config.n_vertices, config.edge_prob, config.weight_range, rng)
        W = make_weight_matrix(edges, config.n_vertices)
        if not _is_acceptable(W, config, registry, penalty_offset):
            continue
        registry.add(weighted_graph(W))
        weights.append(W)
        draw_indices.append(draw)

    shape = (len(weights), config.n_vertices, config.n_vertices)
    return np.asarray(weights, dtype=np.float64).reshape(shape), np.asarray(draw_indices, dtype=np.int64)


def _is_acceptable(
    W: np.ndarray,
    config: InstanceSetConfig,
    registry: _IsomorphismRegistry,
    penalty_offset: float,
) -> bool:
    # Run cheaper checks first (degeneracy -> isomorphic duplicate)
    _, _, assignments = solve_problem_from_weights(W, config.n_vertices, penalty_offset)
    if len(assignments) < config.min_degenerate:
        return False
    return not registry.contains(weighted_graph(W))


def save_instance_set(path: Path, config: InstanceSetConfig, weights: np.ndarray, draw_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W=weights,
        draw_indices=draw_indices,
        generator_version=np.int64(GENERATOR_VERSION),
        n_vertices=np.int64(config.n_vertices),
        edge_prob=np.float64(config.edge_prob),
        weight_range=np.asarray(config.weight_range if config.weight_range is not None else [], dtype=np.float64),
        seed=np.int64(config.seed),
        min_degenerate=np.int64(config.min_degenerate),
    )


def load_instance_set(path: Path) -> tuple[InstanceSetConfig, np.ndarray]:
    """Load a saved instance set and return (config, weights)."""
    with np.load(path) as data:
        version = int(data["generator_version"])
        if version != GENERATOR_VERSION:
            raise ValueError(f"{path} was generated with generator_version={version} (current is {GENERATOR_VERSION})")
        weight_range = tuple(float(v) for v in data["weight_range"]) or None
        config = InstanceSetConfig(
            n_vertices=int(data["n_vertices"]),
            edge_prob=float(data["edge_prob"]),
            weight_range=weight_range,
            seed=int(data["seed"]),
            min_degenerate=int(data["min_degenerate"]),
        )
        return config, data["W"].copy()


def ensure_instance_set(
    config: InstanceSetConfig,
    n_instances: int,
    max_draws: int = 100_000,
    root: Path = INSTANCES_ROOT,
) -> tuple[Path, int]:
    """Ensure the instance set file contains at least n_instances instances and return (path, count contained).

    Existing files are never overwritten. If the existing count is insufficient, the
    set is regenerated, and the extended version is saved only after verifying that its
    leading part exactly matches the existing file (changing an existing instance's index
    would break the correspondence with results computed from it). If n_instances are not
    found within max_draws, only those found are saved.
    """
    path = config.path(root)
    existing = None
    if path.exists():
        existing_config, existing = load_instance_set(path)
        if existing_config != config:
            raise ValueError(f"Generation parameters {existing_config} of {path} do not match the requested {config}")
        if len(existing) >= n_instances:
            return path, len(existing)

    weights, draw_indices = generate_instance_set(config, n_instances, max_draws=max_draws)
    if existing is not None:
        if len(weights) < len(existing) or not np.array_equal(weights[: len(existing)], existing):
            raise RuntimeError(f"Cannot reproduce the existing instances in {path} (the generation rules may have changed)")
        if len(weights) == len(existing):
            return path, len(existing)
    save_instance_set(path, config, weights, draw_indices)
    return path, len(weights)


def load_instance(path: Path, index: int) -> tuple[InstanceSetConfig, np.ndarray]:
    """Extract the index-th weight matrix from an instance set."""
    config, weights = load_instance_set(path)
    if not 0 <= index < len(weights):
        raise IndexError(f"{path} contains only {len(weights)} instance(s) (index={index})")
    return config, weights[index]
