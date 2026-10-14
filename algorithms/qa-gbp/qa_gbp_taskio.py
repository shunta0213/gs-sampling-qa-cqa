"""Utilities that map a parameter dict to its data output directory.

Imported by both qa-gbp.py (the computation) and qa-gbp-add-tasks.py
(task submission), guaranteeing that the same parameters always yield the same
hash and the same directory (same scheme as sa_maxkcut_taskio.py / sqa_maxkcut_taskio.py).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "qa-gbp"

# Parameter name -> type to normalize to. Only parameters that affect the dynamics
# (time evolution) are included. n_steps (number of sesolve time steps) barely affects
# the result, so it is excluded from the hash.
PARAM_TYPES: dict[str, Callable[[Any], Any]] = {
    "W": lambda v: np.asarray(v, dtype=np.float64),
    "n_vertices": int,
    "T": float,
    "penalty": float,
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


def run_dir_for(W: np.ndarray, n_vertices: int, T: float, penalty: float) -> Path:
    """Return the output directory for the run-identifying parameters (does not run anything)."""
    return run_dir(build_params(W=W, n_vertices=n_vertices, T=T, penalty=penalty))
