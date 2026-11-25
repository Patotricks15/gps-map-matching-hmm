# Data

**Source (planned):** Kaggle dataset "Taxi Trajectory Data" (ECML/PKDD 15), slug `crailtap/taxi-trajectory` (540 MB). The plan's Appendix A lists the license as **Unknown**. Confirm it before any redistribution. Nothing from it is committed here.
**Alternative (plan):** T-Drive, `arashnic/tdriver` (Beijing, license "Other"). Not used by the current notebook.
**Read date:** 2026-11-25 (dataset page metadata only, read through the public Kaggle page; no file was downloaded).

## How the data is read

The trajectories are read in place inside a Kaggle Notebook with the dataset attached, under `/kaggle/input/taxi-trajectory/`. Nothing is downloaded to the workstation.

## Road graph (OSM cache)

- Query: `osmnx.graph_from_place("Porto, Portugal", network_type="drive")`, projected to EPSG:32629.
- Cache: `data/cache/porto_drive.graphml`, written on the first run so later runs do not need the network. This cache file is git-ignored with the rest of `data/`.
- City note: the dataset is from Porto, so the graph uses Porto too. The plan's risk table asks for this mismatch to be stated; there is no mismatch here.
