"""Utilities that map a parameter dict to its data output directory.

Imported by both cqa-xy-gbp.py (the computation) and cqa-xy-gbp-add-tasks.py
(task submission), guaranteeing that the same parameters always yield the same
hash and the same directory (same scheme as sa_maxkcut_taskio.py / sqa_maxkcut_taskio.py).
"""

from __future__ import annotations

import pickle
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "cqa-xy-gbp"

# Parameter name -> type to normalize to. Only parameters that affect the dynamics
# (time evolution) are included. n_steps (number of sesolve time steps) barely affects
# the result, so it is excluded from the hash.
# CQA does not use the penalty in its dynamics (H_problem = H_cost only), so the penalty
# is not hashed (so that the same physical run is not split into separate directories
# merely by a different penalty_offset).
PARAM_TYPES: dict[str, Callable[[Any], Any]] = {
    "W": lambda v: np.asarray(v, dtype=np.float64),
    "n_vertices": int,
    "topology": str,
    "T": float,
}


def build_params(**kwargs: Any) -> dict[str, Any]:
    """Build a normalized parameter dict from kwargs (key order fixed to PARAM_TYPES order)."""
    missing = [key for key in PARAM_TYPES if key not in kwargs]
    if missing:
        raise ValueError(f"missing parameter(s): {missing}")
    return {key: caster(kwargs[key]) for key, caster in PARAM_TYPES.items()}


def params_hash(params: dict[str, Any]) -> str:
    hasher = blake2b(digest_size=16)
    hasher.update(pickle.dumps(params))
    return hasher.hexdigest()


def run_dir(params: dict[str, Any]) -> Path:
    return DATA_ROOT / params_hash(params)


def run_dir_for(W: np.ndarray, n_vertices: int, topology: str, T: float) -> Path:
    """Return the output directory for the run-identifying parameters (does not run anything)."""
    return run_dir(build_params(W=W, n_vertices=n_vertices, topology=topology, T=T))
