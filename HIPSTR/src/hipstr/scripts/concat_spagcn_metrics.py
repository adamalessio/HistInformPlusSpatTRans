#!/usr/bin/env python3

"""
Concatenate SpaGCN metrics across seeds + degradation folders.

Expected structure:
spagcn_runs/
  seed_10/
    amb0p4/
      *.spagcn_metrics.csv
    drop0p2/
      *.spagcn_metrics.csv
    ...

Output:
- single CSV with seed + tag parsed from folder structure
"""

from pathlib import Path
import pandas as pd

# =========================
# USER CONFIG
# =========================
BASE_DIR = Path("/home/ilovele1/breast_cancer/spagcn_runs")
OUT_CSV  = Path("/home/ilovele1/breast_cancer/spagcn_all_metrics.csv")

# =========================
# COLLECT FILES
# =========================
rows = []

for seed_dir in sorted(BASE_DIR.glob("seed_*")):
    seed = seed_dir.name.replace("seed_", "")

    for tag_dir in sorted(seed_dir.iterdir()):
        if not tag_dir.is_dir():
            continue

        tag = tag_dir.name

        # find metrics file(s)
        metric_files = list(tag_dir.glob("*.spagcn_metrics.csv"))

        if len(metric_files) == 0:
            print(f"[WARN] No metrics in {tag_dir}")
            continue

        for f in metric_files:
            try:
                df = pd.read_csv(f)

                # add metadata
                df["seed"] = int(seed)
                df["tag"] = tag
                df["source_file"] = str(f)

                rows.append(df)

                print(f"[OK] {f}")

            except Exception as e:
                print(f"[FAIL] {f}: {e}")

# =========================
# CONCAT
# =========================
if len(rows) == 0:
    raise ValueError("No metric files found.")

final_df = pd.concat(rows, ignore_index=True)

# Optional: sort nicely
final_df = final_df.sort_values(
    ["method", "seed", "tag"]
    if "method" in final_df.columns else ["seed", "tag"]
).reset_index(drop=True)

# =========================
# SAVE
# =========================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)

print(f"\n[SAVED] {OUT_CSV}")
print(f"Total rows: {len(final_df)}")