#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end workflow. Replace paths with your local dataset locations.

# 1) Generate fused embeddings / degraded outputs
hipstr-fuse "$@"

# 2) Run evaluation on clustered outputs
hipstr-evaluate "$@"

# 3) Optional: aggregate SpaGCN metrics
hipstr-concat-spagcn

# 4) Recreate manuscript figures
hipstr-plot-breast
hipstr-plot-overlays
