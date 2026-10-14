"""Shared loader for visualize that reads runs from data/qa-gbp and data/cqa-xy-gbp.

There is one cache per data directory (cache/qa-gbp-runs.npz, cache/cqa-xy-gbp-runs.npz)
and one for the merged QA x CQA table (cache/qa-cqa-gbp-merged.pkl), shared by all notebooks.
To change the columns that are loaded, modify the loaders in this file and rebuild with refresh=True.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.run_cache import load_derived_cached, load_runs_cached

REPO_ROOT = Path(__file__).resolve().parents[2]
QA_DIR = REPO_ROOT / "data" / "qa-gbp"
CQA_DIR = REPO_ROOT / "data" / "cqa-xy-gbp"
QA_CACHE_PATH = REPO_ROOT / "cache" / "qa-gbp-runs.npz"
CQA_CACHE_PATH = REPO_ROOT / "cache" / "cqa-xy-gbp-runs.npz"
MERGED_CACHE_PATH = REPO_ROOT / "cache" / "qa-cqa-gbp-merged.pkl"

MERGE_KEYS = ["n_vertices", "seed", "T"]


def _normalize(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    if total <= 0:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) [bit]. Applies additive smoothing and then normalizes, to avoid divergence at zero probabilities."""
    p = _normalize(p) + eps
    q = _normalize(q) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence [bit] (symmetric, within [0, 1], finite even with zero probabilities)."""
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)

    def _kl_safe(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl_safe(p, m) + 0.5 * _kl_safe(q, m)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    return float(0.5 * np.sum(np.abs(p - q)))


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy [bit] of the (unnormalized) probability distribution over degenerate optimal solutions."""
    if probs.size == 0 or probs.sum() == 0:
        return 0.0
    p = probs / probs.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _uniform_distances(probs: np.ndarray) -> dict:
    """Distances from the uniform distribution (i.e. perfectly fair sampling among degenerate optimal solutions)."""
    uniform = np.full(len(probs), 1.0 / len(probs))
    return {
        "entropy": entropy(probs),
        "kl_p_uniform": kl_divergence(probs, uniform),
        "kl_uniform_p": kl_divergence(uniform, probs),
        "js_p_uniform": js_divergence(probs, uniform),
        "tv_p_uniform": total_variation(probs, uniform),
    }


def _load_run(run_dir: Path, algo: Literal["qa", "cqa"]) -> dict | None:
    try:
        input_data = np.load(run_dir / "input.npz")
        output_data = np.load(run_dir / "output.npz")
    except FileNotFoundError:
        return None
    probs = output_data["optimal_partition_probs"]
    row = {
        "run_id": run_dir.name,
        "seed": int(input_data["seed"]),
        "edge_prob": float(input_data["edge_prob"]),
        "n_vertices": int(input_data["n_vertices"]),
        "T": float(input_data["T"]),
        "n_steps": int(input_data["n_steps"]),
        "penalty": float(input_data["penalty"]),
    }
    if algo == "qa":
        row["penalty_offset"] = float(input_data["penalty_offset"])
    else:
        row["topology"] = str(input_data["topology"])
    row |= {
        "n_degenerate": len(probs),
        "feasible_probability": float(output_data["feasible_probability"]),
        "optimal_probability": float(output_data["optimal_probability"]),
        "elapsed_time": float(output_data["elapsed_time"]),
        "optimal_partitions": output_data["optimal_partitions"],
        "optimal_partition_probs": probs,
    }
    return row | _uniform_distances(probs)


def load_qa_run(run_dir: Path) -> dict | None:
    return _load_run(run_dir, "qa")


def load_cqa_run(run_dir: Path) -> dict | None:
    return _load_run(run_dir, "cqa")


def _check_columns(df: pd.DataFrame, data_dir: Path, loader, cache_path: Path) -> None:
    """Check that the cache columns match the current loader's columns (to avoid silently using a stale cache)."""
    if df.empty:
        return
    expected = loader(data_dir / df["run_id"].iloc[0])
    if expected is not None and set(expected) != set(df.columns):
        raise RuntimeError(f"Columns of {cache_path} do not match the loader. Rebuild with refresh=True")


def load_qa_runs(refresh: bool | Literal["new"] = False) -> pd.DataFrame:
    df = load_runs_cached(QA_DIR, load_qa_run, QA_CACHE_PATH, refresh=refresh)
    _check_columns(df, QA_DIR, load_qa_run, QA_CACHE_PATH)
    return df


def load_cqa_runs(refresh: bool | Literal["new"] = False) -> pd.DataFrame:
    df = load_runs_cached(CQA_DIR, load_cqa_run, CQA_CACHE_PATH, refresh=refresh)
    _check_columns(df, CQA_DIR, load_cqa_run, CQA_CACHE_PATH)
    return df


def _merge_qa_cqa(df_qa: pd.DataFrame, df_cqa: pd.DataFrame) -> pd.DataFrame:
    """Join QA and CQA runs with the same (n_vertices, seed, T) and add distances between their distributions.

    Columns present in both QA and CQA get the _qa / _cqa suffix (topology and penalty_offset exist in only one, so they keep their names).
    """
    merged = df_qa.merge(df_cqa, on=MERGE_KEYS, suffixes=("_qa", "_cqa"))
    n_mismatch = sum(
        not np.array_equal(a, b) for a, b in zip(merged["optimal_partitions_qa"], merged["optimal_partitions_cqa"])
    )
    assert n_mismatch == 0, (
        "Different optimal_partitions obtained from the same seed (problem generation parameters may be inconsistent)"
    )
    # optimal_partitions is no longer needed after validation, so drop it to keep the cache small
    merged = merged.drop(columns=["optimal_partitions_qa", "optimal_partitions_cqa"])

    p_cqa, p_qa = merged["optimal_partition_probs_cqa"], merged["optimal_partition_probs_qa"]
    merged["kl_cqa_qa"] = [kl_divergence(p, q) for p, q in zip(p_cqa, p_qa)]
    merged["kl_qa_cqa"] = [kl_divergence(q, p) for p, q in zip(p_cqa, p_qa)]
    merged["js"] = [js_divergence(p, q) for p, q in zip(p_cqa, p_qa)]
    merged["tv"] = [total_variation(p, q) for p, q in zip(p_cqa, p_qa)]
    return merged


def load_qa_cqa_merged(df_qa: pd.DataFrame, df_cqa: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """DataFrame joining the results of load_qa_runs / load_cqa_runs (rebuilt automatically when the run caches are updated)."""
    return load_derived_cached(
        MERGED_CACHE_PATH,
        lambda: _merge_qa_cqa(df_qa, df_cqa),
        depends_on=[QA_CACHE_PATH, CQA_CACHE_PATH],
        refresh=refresh,
    )
