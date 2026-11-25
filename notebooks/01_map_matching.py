# # RT03 - How do we snap noisy GPS points to the road graph?
#
# Question: how do we snap noisy GPS points to the road graph?
#
# Data: Porto taxi trajectories, ECML/PKDD 15, Kaggle `crailtap/taxi-trajectory`, read in place from `/kaggle/input/taxi-trajectory/`.
# Ground truth: the recorded POLYLINE of each trip. Observations: the same polyline with Gaussian noise added (sigma 10 m).
# Road graph: OpenStreetMap drive network for Porto, projected to EPSG:32629 (UTM 29N), cached to `data/cache/porto_drive.graphml`.
# Method: HMM with Gaussian emission on distance to edge and exponential transition on |route - great-circle| distance, decoded by Viterbi.
# Baseline: nearest-edge snapping per fix.

# %%
import glob
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from map_matching import mean_error_m, nearest_edge_baseline, viterbi_match
from map_matching.geometry import edge_length

INPUT_DIR = Path("/kaggle/input/taxi-trajectory")
CACHE = Path("../data/cache/porto_drive.graphml")
FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
CITY = "Porto, Portugal"
UTM_CRS = "EPSG:32629"
MODEL = "RT03 v0.1.0"
SIGMA_M = 20.0  # emission scale for the HMM
BETA_M = 10.0  # transition scale for the HMM
RADIUS_M = 60.0  # candidate search radius
NOISE_M = 10.0  # Gaussian noise added to the recorded polyline
N_TRIPS = 50
RNG = np.random.default_rng(42)

# %% [markdown]
# ## 1. Load the road graph (cached)
#
# The OSM graph is cached to GraphML so runs do not need the network. It is projected to metres, and each node gets `x`/`y`.

# %%
if CACHE.exists():
    G_multi = ox.load_graphml(CACHE)
else:
    G_multi = ox.graph_from_place(CITY, network_type="drive")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_multi, CACHE)
G_utm = ox.project_graph(G_multi, to_crs=UTM_CRS)
G = nx.DiGraph(G_utm)  # one edge per node pair; parallel edges are collapsed
for n, d in G.nodes(data=True):
    d["x"], d["y"] = float(d["x"]), float(d["y"])
print(f"{MODEL} | graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# %% [markdown]
# ## 2. Load trajectories with the planned schema check
#
# The ECML/PKDD train file has a `POLYLINE` column: a JSON list of `[lon, lat]` points at 15 s intervals.

# %%
files = sorted(glob.glob(str(INPUT_DIR / "**" / "*.csv"), recursive=True))
print("CSV files:", files)
trips_raw = pd.read_csv(files[0], usecols=["TRIP_ID", "POLYLINE"])
assert "POLYLINE" in trips_raw.columns, "schema check failed: POLYLINE column missing"
trips_raw["polyline"] = trips_raw["POLYLINE"].map(json.loads)
trips_raw = trips_raw[trips_raw["polyline"].map(len) >= 10].sample(N_TRIPS, random_state=42)

# %% [markdown]
# ## 3. Match each trip with the HMM and with the nearest-edge baseline

# %%
rows = []
for _, trip in trips_raw.iterrows():
    lonlat = np.array(trip["polyline"], dtype=float)
    truth_xy = gpd.GeoSeries(gpd.points_from_xy(lonlat[:, 0], lonlat[:, 1]), crs="EPSG:4326").to_crs(UTM_CRS).get_coordinates().to_numpy()
    obs_xy = truth_xy + RNG.normal(0, NOISE_M, truth_xy.shape)
    # Restrict the graph to nodes near this trip so candidate search stays fast.
    xmin, ymin = truth_xy.min(axis=0) - 500
    xmax, ymax = truth_xy.max(axis=0) + 500
    nodes = [n for n, d in G.nodes(data=True) if xmin <= d["x"] <= xmax and ymin <= d["y"] <= ymax]
    sub = G.subgraph(nodes).copy()
    try:
        hmm = viterbi_match(sub, obs_xy, sigma=SIGMA_M, beta=BETA_M, radius=RADIUS_M)
        base = nearest_edge_baseline(sub, obs_xy, radius=RADIUS_M)
    except ValueError:
        continue  # a fix had no road candidate inside the search radius
    rows.append({
        "trip_id": trip["TRIP_ID"],
        "hmm_error_m": mean_error_m(hmm, truth_xy),
        "nearest_error_m": mean_error_m(base, truth_xy),
        "hmm_path_m": sum(edge_length(sub, a.u, a.v) for a in hmm),
    })
results = pd.DataFrame(rows)
print(results.describe().round(2))

# %% [markdown]
# ## 4. Results and figure
#
# The figure compares per-trip mean error for the two methods across the sampled trips. The title includes the city, model version, and noise level.

# %%
summary = results[["hmm_error_m", "nearest_error_m"]].agg(["mean", "std"]).round(2)
print(summary)
fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([results["nearest_error_m"], results["hmm_error_m"]], tick_labels=["nearest edge", "HMM"])
ax.set_ylabel("mean error to truth (m)")
ax.set_title(f"GPS error by method | {CITY} | noise {NOISE_M:.0f} m | {MODEL}")
fig.tight_layout()
fig.savefig(FIG_DIR / "rt03_error_by_method.png", dpi=150)
plt.show()
