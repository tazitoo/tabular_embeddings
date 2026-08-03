# Reproducibility: what is bit-identical, and what is not

Established 2026-08-02/03 by rebuilding the worker fleet onto a single locked
environment and measuring, not by reasoning about it. Reproduce any claim here
with `scripts/rebuttal/carte_determinism_check.py`.

## The short version

| scope | result |
|---|---|
| same host, same GPU, repeated run | **bit-identical**, all 6 models |
| surfer / terrax / octo (3× RTX 3090) | **bit-identical** to each other, all 5 models tested |
| firelord (RTX 4090) vs the 3090 pool | **differs**, all models — GPU architecture |
| morg (RTX 3090, but EPYC + driver 595) | **differs for tabpfn only** |

The 3× 3090 pool is the bit-reproducible set. Pin reproducibility-critical runs
there.

## Required for within-host determinism

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8   # env, before CUDA init
torch.use_deterministic_algorithms(True)
```

Seeding alone is NOT sufficient and never was. Every RNG in the path is already
seeded (`functional_decomposition.py` sets `manual_seed`/`np.random.seed`;
`CARTEClassifier` defaults to `random_state=0` and re-seeds torch internally,
overriding ours). The full standard recipe — python/numpy/torch RNGs, both CUDA
seeders, `cudnn.deterministic`, `cudnn.benchmark=False` — changes nothing
measurable. `cudnn.*` governs *convolution* algorithm selection and CARTE has no
convolutions; the operative knob is `use_deterministic_algorithms`, which covers
the scatter/index_add ops PyG uses for graph aggregation.

Enabling it does **not** perturb the already-deterministic models: cross-mode
comparison showed mitra/tabdpt/tabicl/tabicl_v2/tabpfn byte-for-byte unchanged.
So it is safe to enable globally.

### Why only CARTE needed this

CARTE is the only recipient whose tail is **trained** rather than loaded —
`CARTETail.from_data` calls `clf.fit(...)` for up to `max_epoch` epochs with an
internal val split and best-epoch weight restoration (`copy.deepcopy` on every
improvement). The other five load pretrained weights and run inference only.
Without the determinism flag, carte's baseline predictions differed on **86/86
rows** between two builds in the *same process*; tabpfn was bit-identical in the
same test.

Consequence: carte is also the most hardware- and dependency-sensitive model,
for the same structural reason. Its cross-host divergence (~5e-02) is roughly
the same size whatever differs, consistent with chaotic amplification through
training.

## What does NOT affect results

- **Minor driver versions.** surfer (580.126.09) vs octo (580.173.02), same GPU
  and software: bit-identical, all 5 models. Does not extend to major versions
  — see morg below.
- **scikit-learn version.** terrax (1.7.2) vs octo (1.6.1), all else equal: the
  four pretrained models are bit-identical. Only carte differs, because
  `Table2GraphTransformer` runs `PowerTransformer` live on every build while the
  other models read preprocessing from cache.
- **Thread count.** Pinning `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` did not
  change morg's tabpfn deviation by a single digit.
- **Most transitive dependencies.** 26 of the 65 that once differed across
  workers are never imported during compute at all (accelerate, boto3, optuna,
  plotly, sqlalchemy, ...). Verified with
  `scripts/rebuttal/compute_path_modules.py`. Note 39 *are* imported — but
  `numexpr`/`bottleneck` are imported and demonstrably inert (terrax lacks them
  entirely and still matches), so "imported" does not imply "matters".

## What DOES affect results

- **GPU architecture.** firelord (4090) vs any 3090, identical software and
  determinism enabled: carte 4.75e-02, mitra 2.48e-04, tabdpt 1.84e-04,
  tabpfn 6.89e-06, tabicl 3.76e-06 (mean |Δ| on baseline predictions, `anneal`).
  Exactly reproducible across repeats.
- **morg, for tabpfn only** (2.90e-04 mean, 1.84e-02 max). morg differs from the
  3090 pool in BOTH driver major version (595.71.05 vs 580.173.02) and CPU
  (EPYC 7542 32-core vs Ryzen 7 3700X 8-core). **UNRESOLVED** — the pool has no
  host with 595+Ryzen or 580+EPYC to separate them. Deterministic (identical to
  the digit across repeats and thread settings), so it is a systematic code-path
  difference, not noise.

### Scope of these magnitudes — read before quoting them

All figures above are **baseline predictions on one dataset (`anneal`, 86 rows)**.
They are NOT bounds, and they say nothing directly about the reported quantities.
`gc = (loss_weak − loss_int)/(loss_weak − loss_strong)` divides by a per-row gap
that is small on near-tie rows, so a tiny prediction change can move `gc` a lot.
There is direct evidence the relationship is not simple: tabdpt's cross-host
prediction difference is ~1.8e-04, yet its aggregate `rel_on` moved 0.005–0.013
between two decomposition runs. Unreconciled. To bound the effect on reported
numbers, run the same decomposition pairs on a 3090 and on firelord and diff the
`gc` aggregates — not the predictions.

## Environment management

`pyproject.toml` pins direct dependencies AND python (`>=3.12,<3.13`).
`uv.lock` pins the full transitive tree (176 packages) including torch's
`+cu124` build via a declared index. Rebuild a worker with:

```bash
conda create -y -n tfm python=3.12.12
cd /home/brian/src/tabular_embeddings
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --extra all --inexact
```

`--inexact` preserves host-local tooling (ipython, ruff, ...) outside the lock;
use `--exact` for a byte-identical tree, at the cost of removing those.

`tfm2` (tabicl_v2) is a **separate environment**, not an extra — it pins
torch 2.7.1 against the root's 2.6.0 and the two are mutually unsatisfiable.
See `envs/tfm2/pyproject.toml`.

### How the drift happened

`requires-python = ">=3.10"` and floors like `numpy>=1.24` let each env resolve
to whatever was current on its build date. surfer and morg landed on python 3.10,
which *caps* numpy at 2.2.x — so the three-way numpy split (1.26.4 / 2.2.6 /
2.3.5) was a symptom of the python floor, not an independent problem.
