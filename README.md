This repository contains a Hierarchial QAOA-based pipeline with local feasibility repair for a vehicle routing problem. The example problem uses 12 customer points, clusters them into three groups of four, solves an open TSP (OTSP) inside each cluster, solves an inter-cluster VRP (ICVRP) over the depot and cluster centroids, post-processes sampled bitstrings based on a local-bit flip search, and merges the pieces into full routes.

The correspoding article can be found at: https://arxiv.org/abs/2511.00506

## Repository Layout

- `solver.py` - one-command runner for the example pipeline.
- `OTSP.py` - OTSP QUBO construction and QAOA samplers for cluster-level routes.
- `ICVRP.py` - ICVRP QUBO construction and QAOA samplers for depot/centroid routing.
- `scripts/Preprocess.py` - constrained k-means clustering and entry/exit point selection.
- `post_process.py` - original ICVRP post-processing helpers.
- `post_process_otsp.py` - original OTSP post-processing helpers.
- `merge_otsp_icvrp.py` - combines post-processed OTSP and ICVRP bitstrings into final customer routes.
- `data_example/` - small example datasets, `dataset0.pkl` through `dataset9.pkl`.
- `datagen.py` - utility for generating random 12-customer examples.

## Setup

Install dependencies:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Qiskit APIs used here expect the versions pinned in `requirements.txt`.

## Run The Example Pipeline

From the repository root:

```
python solver.py --data data_example/dataset1.pkl --data_id 1 --run_seed 0 --p 3 --prob 1 --variant std
```

Arguments:

- `--data` - dataset pickle path, default `data_example/dataset1.pkl`.
- `--data_id` - dataset id used in output filenames, default `1`.
- `--run_seed` - clustering/QAOA seed, default `0`.
- `--p` - QAOA depth, default `3`.
- `--prob` - probability threshold tag; `1` means `0.001`.
- `--variant` - merged variant to report, either `std` or `ma`, default `std`.

The runner writes raw, post-processed, and merged outputs:


## Pipeline Summary

1. Load a 12-point customer dataset.
2. Cluster customers into three equal-size clusters using constrained k-means.
3. For each cluster, solve OTSP from selected entry to exit points.
4. Build a 4-node ICVRP instance using depot plus three centroids.
5. Run standard QAOA and MA-QAOA samplers.
6. Post-process sampled bitstrings into feasible candidates.
7. Merge ICVRP cluster order with OTSP intra-cluster paths.


## Note: 
Shreetam Dash, SoA University has contributed equally in developing this codebase.
