# unfair-sampling-qa-cqa-gbp

Code accompanying the paper *[Ground-State Sampling Correspondence between Penalty-Based and Constrained Quantum Annealing](https://arxiv.org/abs/2610.26687)* (Shunta Ide and Shu Tanaka, arXiv:2610.26687).

It generates the simulation data of

- **QA**: quantum annealing with the transverse-field (X) mixer and a penalty term for the balance constraint, and
- **CQA**: constraint-preserving quantum annealing with the XY mixer on the complete graph,

applied to random weighted graph bipartitioning (GBP) instances, and produces all figures in the paper from these data.

The raw simulation data are not included in this repository; they can be regenerated with the scripts below
(see [Data availability](#data-availability)).

## Repository layout

| Path | Contents |
|---|---|
| `src/gbp/` | GBP instance generation, Hamiltonians, drivers, exact solver, loading of run outputs |
| `src/maxkcut/` | Shared graph/weight utilities used by `src/gbp/` |
| `src/run_cache.py` | Caching of the per-run outputs into a single table |
| `algorithms/qa-gbp/` | QA simulation (`qa-gbp.py`) and the parameter sweep used in the paper (`qa-gbp-add-tasks.py`) |
| `algorithms/cqa-xy-gbp/` | CQA simulation (`cqa-xy-gbp.py`) and the parameter sweep used in the paper (`cqa-xy-gbp-add-tasks.py`) |
| `visualize/journal.ipynb` | Analysis and all figures in the paper |
| `tests/` | Unit tests |

Comments and docstrings in the code are written in Japanese.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- [pueue](https://github.com/Nukesor/pueue) (optional; used to run the parameter sweep in parallel)
- A LaTeX installation (figures are rendered with `text.usetex`)

Install the dependencies (versions are pinned in `uv.lock`):

```bash
uv sync
uv run pytest   # optional
```

## Reproducing the results

### 1. Generate the simulation data

Each script `*-add-tasks.py` enumerates the parameter grid used in the paper and runs one simulation per parameter set.
The parameters are set as constants at the top of each script:

| | QA (`qa-gbp-add-tasks.py`) | CQA (`cqa-xy-gbp-add-tasks.py`) |
|---|---|---|
| Number of vertices $N$ | 6, 8, 10, 12 | 6, 8, 10, 12 |
| Instances (seeds) | 0–99 | 0–99 |
| Edge probability | 0.8 | 0.8 |
| Annealing time $T$ | $\{1, 1.5, 2, 3, 5, 7\} \times 10^k$, $1 \le T \le 10^5$ | same |
| Penalty offset $\mu_+$ | $\{1, 1.5, 2, 3, 5, 7\} \times 10^k$, $0.01 \le \mu_+ \le 10$ | – |
| Time steps / rtol / atol | $10^5$ / $10^{-10}$ / $10^{-12}$ | same |

```bash
# with pueue (set PARALLEL in the script to the number of available cores)
pueued -d
uv run python algorithms/qa-gbp/qa-gbp-add-tasks.py
uv run python algorithms/cqa-xy-gbp/cqa-xy-gbp-add-tasks.py
```

Set `USE_PUEUE = False` in the scripts to run the tasks serially without pueue, or `DRY_RUN = True` to print the commands only.
A single run can also be executed directly, e.g.

```bash
uv run python algorithms/qa-gbp/qa-gbp.py --n-vertices 8 --seed 0 --time 100 --penalty-offset 0.1
```

The outputs are written to `data/qa-gbp/<hash>/` and `data/cqa-xy-gbp/<hash>/`, where `<hash>` is determined by the run parameters.
Instances are generated deterministically from the seed (`src/gbp/search.py`), so QA and CQA runs with the same seed use the same graph.

### 2. Make the figures

Run `visualize/journal.ipynb` (with `visualize/` as the working directory).
On the first run, the outputs in `data/` are collected into `cache/`, and the figures are written to `visualize/figs/journal/`.

## Data availability

The raw simulation data underlying the figures are available from the corresponding author upon reasonable request.
They can also be regenerated with the code in this repository as described above.

## Citation

If you use this code, please cite the following paper:

> S. Ide and S. Tanaka, "Ground-State Sampling Correspondence between Penalty-Based and Constrained Quantum Annealing," arXiv:2610.26687 (2026).

```bibtex
@misc{ide2026groundstate,
  title         = {Ground-State Sampling Correspondence between Penalty-Based and Constrained Quantum Annealing},
  author        = {Ide, Shunta and Tanaka, Shu},
  year          = {2026},
  eprint        = {2610.26687},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2610.26687},
  url           = {https://arxiv.org/abs/2610.26687}
}
```
