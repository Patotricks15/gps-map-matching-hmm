# gps-map-matching-hmm (RT03)

**Question:** how do we snap noisy GPS points to the road graph?

## Method

- Road graph: OSM drive network for Porto, projected to metres (EPSG:32629), with directed edges.
- Candidates: each fix is matched to every edge within 60 m of its projection.
- Emission: Gaussian on the distance from the fix to the candidate point, sigma = 20 m.
- Transition: exponential on `|route distance - great-circle distance|` between consecutive fixes, beta = 10 m. Route distance uses Dijkstra on the directed graph.
- Decoding: Viterbi in log space (`map_matching.hmm.viterbi_match`).
- Baseline: nearest edge per fix, with no transitions.
- Metric: mean distance in metres between the matched road point and the ground-truth position.

## Evaluation design

The dataset gives the recorded GPS polyline, not a road-matched path. The ground truth is therefore the recorded polyline. The observations are that polyline with Gaussian noise (sigma 10 m, seed 42) added. This is a controlled noise test on real trajectories, not a synthetic dataset.

## Status

- **Notebook not executed.** `notebooks/01_map_matching.py` (and its `.ipynb`) was written but not run locally, because the data is not on this machine. It runs in a Kaggle Notebook with the dataset attached. The OSM download needs network access.
- No figures are committed yet. The notebook writes `reports/figures/rt03_error_by_method.png` when it runs.
- No matched-error numbers are reported yet.

## Limitations

- The ECML/PKDD trajectories are sampled every 15 s, so the path between fixes is uncertain. Road-level truth is not available.
- Candidate search is a linear scan over the edges of a trip-sized subgraph. It is fine for a sample of trips but does not scale to a city-wide batch.
- The directed graph keeps one edge per node pair, so parallel OSM edges are collapsed.

## Tests

```bash
make install
make test
```

Tests use a hand-built two-road fixture (`tests/test_hmm.py`). They cover the Gaussian emission formula and its peak, the transition preference, route distance equal to Dijkstra on the graph, Viterbi keeping the consistent road where nearest snapping jumps, and the candidate radius.

## Reproduce

Attach the Kaggle dataset in a Kaggle Notebook, upload this repo's `src/` package, and run `notebooks/01_map_matching.ipynb` with Run All. Locally, `make notebook` only works if the dataset is present under `/kaggle/input/taxi-trajectory`.
