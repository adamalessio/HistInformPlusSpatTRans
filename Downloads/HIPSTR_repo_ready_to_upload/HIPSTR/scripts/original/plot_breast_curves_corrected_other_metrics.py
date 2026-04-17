#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# USER INPUT
# =========================
df4_path = "/home/ilovele1/Brain/baselines_1042/Breast_grouped_metrics_with_seeds_4clusters.csv"
df20_path = "/home/ilovele1/Brain/baselines_1042/Breast_grouped_metrics_with_seeds_20clusters.csv"
sp_path = "/home/ilovele1/breast_cancer/Breast_spagcn_all_metrics.csv"

outdir = Path("breast_cluster_curves_corrected_axes")
outdir.mkdir(exist_ok=True)

# =========================
# LOAD + PREP
# =========================
def prep(df):
    return df.rename(columns={
        "ARI_vs_truth": "ARI_vs_pathologist",
        "NMI_vs_truth": "NMI_vs_pathologist",
        "kNN_Jaccard_vs_baseline": "kNN_Jaccard",
    })

df4 = prep(pd.read_csv(df4_path))
df20 = prep(pd.read_csv(df20_path))
sp = pd.read_csv(sp_path)

def family_from_tag(tag):
    tag = str(tag)
    if tag.startswith("amb"):
        return "ambient"
    if tag.startswith("drop"):
        return "dropout"
    if tag.startswith("cap"):
        return "capture"
    return "other"

def spagcn_label_type_from_source(path):
    path = str(path)

    # 20-cluster runs look like seed_20_0, seed_20_1, ...
    if "/seed_20_" in path:
        return "20clusters"

    # 4-cluster runs look like seed_6, seed_7, ..., seed_10
    # adjust this if you later add other seed naming conventions
    m = re.search(r"/seed_(\d+)/", path)
    if m:
        return "4clusters"

    return None

for df in [df4, df20, sp]:
    df["family"] = df["tag"].apply(family_from_tag)

# harmonize spaGCN naming if needed
if "method" not in sp.columns:
    sp["method"] = "spagcn"

if "source_file" not in sp.columns:
    raise ValueError("SpaGCN CSV must contain a source_file column to separate 4- vs 20-cluster runs.")

sp["label_type"] = sp["source_file"].apply(spagcn_label_type_from_source)

sp4 = sp[sp["label_type"] == "4clusters"].copy()
sp20 = sp[sp["label_type"] == "20clusters"].copy()

print("SpaGCN 4-cluster rows:", len(sp4))
print("SpaGCN 20-cluster rows:", len(sp20))

comb4 = pd.concat([df4, sp4], ignore_index=True, sort=False)
comb20 = pd.concat([df20, sp20], ignore_index=True, sort=False)
method_display = {
    "RNA": "ST-Only",
    "concat": "HIPSTR-Concat",
    "early_ae": "HIPSTR-Early",
    "mid_ae": "HIPSTR-Mid",
    "late_ae": "HIPSTR-Late",
    "clip": "HIPSTR-CLIP",
    "radiomics": "Hist-Only",
    "Radiomics": "Hist-Only",
    "spagcn": "SpaGCN",
}

# =========================
# AXIS ORDER
# =========================
def tag_order(family):
    if family == "ambient":
        return ["amb0", "amb0p2", "amb0p4", "amb0p6", "amb0p8"]
    if family == "dropout":
        return ["amb0", "drop0p2", "drop0p4", "drop0p6", "drop0p8"]
    if family == "capture":
        return ["amb0", "cap0p1", "cap0p3", "cap0p5", "cap0p7", "cap0p9"]
    return []

def summarize(df, metric, family):
    sub = df[df["family"] == family].copy()
    if metric not in sub.columns or sub.empty:
        return None

    order = tag_order(family)
    sub = sub[sub["tag"].isin(order)].copy()
    if sub.empty:
        return None

    sub["tag"] = pd.Categorical(sub["tag"], categories=order, ordered=True)
    g = sub.groupby(["method", "tag"], observed=True)[metric].agg(["mean", "std"]).reset_index()
    return g

# =========================
# METRIC GROUPS
# =========================
core_metrics = [
    ("ARI_vs_baseline", "ARI vs baseline"),
    ("NMI_vs_baseline", "NMI vs baseline"),
    ("ARI_vs_pathologist", "ARI vs pathologist"),
    ("NMI_vs_pathologist", "NMI vs pathologist"),
]

other_metrics = [
    ("kNN_Jaccard", "kNN Jaccard vs baseline"),
    ("fragmentation_index", "Fragmentation index"),
    ("merging_index", "Merging index"),
    ("cluster_persistence", "Cluster persistence"),
    ("run_moran_i", "Run Moran's I"),
    ("run_geary_c", "Run Geary's C"),
    ("entropy_run", "Entropy"),
    ("gini_run", "Gini"),
    ("n_clusters_run", "Number of clusters"),
]

# =========================
# PLOTTING HELPERS
# =========================
def add_single_legend(fig, axes):
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(labels)),
                frameon=False,
                fontsize=10
            )
            return

def plot_metric_grid(df, label_name, family, metrics, filename,  ncols=3):
    present = [m for m in metrics if m[0] in df.columns]
    if not present:
        return

    order = tag_order(family)
    methods = sorted(df["method"].dropna().astype(str).unique())

    n = len(present)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax, (metric, title) in zip(axes.flatten(), present):
        s = summarize(df, metric, family)
        if s is None:
            ax.set_visible(False)
            continue

        for method in methods:
            sub = s[s["method"] == method].copy()
            if sub.empty:
                continue

            x = np.arange(len(sub))
            ax.errorbar(
                x,
                sub["mean"],
                yerr=sub["std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                label=method_display.get(method, method),
            )

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Condition")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax in axes.flatten()[len(present):]:
        ax.set_visible(False)

    add_single_legend(fig, axes)
    #fig.suptitle(f"{fig_title}: {label_name}, {family}", fontsize=18)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    png = outdir / f"{filename}.png"
    pdf = outdir / f"{filename}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

# =========================
# RUN
# =========================
for label_name, df in [("4clusters", comb4),("20clusters",comb20)]:
    for family in ["ambient", "dropout", "capture"]:
        plot_metric_grid(
            df,
            label_name,
            family,
            core_metrics,
            f"breast_{label_name}_{family}_core_metrics_corrected_axes",
            #"Breast cancer core metrics",
            ncols=2,
        )

        plot_metric_grid(
            df,
            label_name,
            family,
            other_metrics,
            f"breast_{label_name}_{family}_other_metrics_corrected_axes",
            #"Breast cancer additional metrics",
            ncols=3,
        )

print("Done. Figures saved to:", outdir)