#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

FILENAME_RE = re.compile(
    r"(.+?)_((?:amb|cap|drop)[0-9p]+(?:__(?:amb|cap|drop)[0-9p]+)?)_seed([0-9]+)\.clustered_([0-9]+)clusters\.h5ad$"
)


def parse_method_tag_seed(filename: str):
    name = Path(filename).name
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Could not parse method/tag/seed from filename: {name}")
    method = m.group(1)
    tag = m.group(2)
    seed = int(m.group(3))
    n_clusters = int(m.group(4))
    return method, tag, seed, n_clusters


def find_h5ads(input_dir: str, recursive: bool = True) -> pd.DataFrame:
    p = Path(input_dir)
    files = sorted(p.rglob("*.h5ad") if recursive else p.glob("*.h5ad"))
    rows = []

    for fp in files:
        if not fp.is_file():
            continue

        name = fp.name

        # Standard pattern
        try:
            method, tag, seed, n_clusters = parse_method_tag_seed(name)
            rows.append({
                "file": str(fp),
                "method": method,
                "tag": tag,
                "seed": seed,
                "target_clusters_from_name": n_clusters,
            })
            continue
        except Exception:
            pass

        # SpaGCN fallback
        try:
            if name.endswith(".spagcn.h5ad"):
                parts = fp.parts
                seed_parts = [part for part in parts if part.startswith("seed_")]
                if not seed_parts:
                    continue
                seed_token = seed_parts[-1].replace("seed_", "")
                seed = int(seed_token.split("_")[-1])  # handles seed_7_7
                tag = fp.parent.name
                rows.append({
                    "file": str(fp),
                    "method": "spagcn",
                    "tag": tag,
                    "seed": seed,
                    "target_clusters_from_name": None,
                })
        except Exception:
            pass

    return pd.DataFrame(rows)


def method_display(method: str) -> str:
    mapping = {
        "RNA": "ST-Only",
        "rna": "ST-Only",
        "concat": "HIPSTR-Concat",
        "early_ae": "HIPSTR-Early",
        "mid_ae": "HIPSTR-Mid",
        "late_ae": "HIPSTR-Late",
        "clip": "HIPSTR-CLIP",
        "radiomics": "Hist-Only",
        "spagcn": "SpaGCN",
    }
    return mapping.get(method, method)


def auto_pick_cluster_key(adata: ad.AnnData, preferred: Optional[str] = None) -> str:
    if preferred is not None and preferred in adata.obs.columns:
        return preferred

    cols = list(map(str, adata.obs.columns))
    for c in ["clusters", "cluster", "leiden", "refined_pred", "pred"]:
        if c in cols:
            return c

    leiden_like = [c for c in cols if "leiden" in c.lower()]
    if len(leiden_like) > 0:
        leiden_like = sorted(leiden_like, key=lambda x: (len(x), x), reverse=True)
        return leiden_like[0]

    raise KeyError(f"Could not infer cluster key from .obs columns: {cols[:20]}")


def extract_coords_from_adata(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs

    # Prefer full-resolution pixel coordinates
    for a, b in [
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("pxl_col", "pxl_row"),
        ("x", "y"),
    ]:
        if a in obs.columns and b in obs.columns:
            df = pd.DataFrame({
                "barcode": canon_barcodes(pd.Index(adata.obs_names)),
                "x": obs[a].to_numpy(dtype=float, copy=True),
                "y": obs[b].to_numpy(dtype=float, copy=True),
            })
            return df.drop_duplicates("barcode").set_index("barcode")

    # Then try obsm["spatial"]
    if "spatial" in adata.obsm:
        X = np.asarray(adata.obsm["spatial"])
        if X.ndim == 2 and X.shape[1] >= 2:
            df = pd.DataFrame({
                "barcode": canon_barcodes(pd.Index(adata.obs_names)),
                "x": X[:, 0].astype(float, copy=False),
                "y": X[:, 1].astype(float, copy=False),
            })
            return df.drop_duplicates("barcode").set_index("barcode")

    # Last resort: array coords
    for a, b in [
        ("array_col", "array_row"),
        ("col", "row"),
    ]:
        if a in obs.columns and b in obs.columns:
            df = pd.DataFrame({
                "barcode": canon_barcodes(pd.Index(adata.obs_names)),
                "x": obs[a].to_numpy(dtype=float, copy=True),
                "y": obs[b].to_numpy(dtype=float, copy=True),
            })
            return df.drop_duplicates("barcode").set_index("barcode")

    raise ValueError("No usable spatial coordinates found in reference adata.")


def build_reference_coords(
    file_table: pd.DataFrame,
    seed: int,
    ref_method: str = "RNA",
    ref_tag: str = "amb0",
) -> pd.DataFrame:
    sub = file_table[
        (file_table["method"] == ref_method) &
        (file_table["tag"] == ref_tag) &
        (file_table["seed"] == seed)
    ]
    if sub.empty:
        raise ValueError(
            f"No reference file found for method={ref_method}, tag={ref_tag}, seed={seed}"
        )

    ref_file = sub.iloc[0]["file"]
    ref_adata = ad.read_h5ad(ref_file)
    return extract_coords_from_adata(ref_adata)


def align_labels_to_reference(
    adata_run: ad.AnnData,
    cluster_key: str,
    ref_coords: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    run_bc = canon_barcodes(pd.Index(adata_run.obs_names))
    run_labels = pd.Series(
        adata_run.obs[cluster_key].astype(str).to_numpy(),
        index=run_bc,
    )
    run_labels = run_labels[~run_labels.index.duplicated(keep="first")]

    common = ref_coords.index.intersection(run_labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between run and reference coordinates.")

    coords = ref_coords.loc[common]
    labels = run_labels.loc[common]
    return coords, labels

def canon_barcodes(idx: pd.Index) -> pd.Index:
    s = idx.astype(str).str.strip()
    s = s.str.replace(r"\.\d+$", "", regex=True)
    s = s.str.replace(r"^.*_([ACGT]+-[0-9]+)$", r"\1", regex=True)
    return pd.Index(s)


def match_clusters_to_baseline(base_labels: np.ndarray, run_labels: np.ndarray) -> dict[str, str]:
    """
    Map run-cluster -> baseline-cluster by maximal overlap.
    """
    ct = pd.crosstab(
        pd.Series(base_labels.astype(str), name="baseline"),
        pd.Series(run_labels.astype(str), name="run")
    )

    mapping = {}
    for run_clust in ct.columns:
        col = ct[run_clust]
        if col.sum() == 0:
            continue
        base_clust = col.idxmax()
        mapping[str(run_clust)] = str(base_clust)

    return mapping


def build_baseline_palette(base_labels: list[str]) -> dict[str, tuple]:
    uniq = sorted(pd.unique(base_labels))
    cmap = plt.get_cmap("tab20")
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def align_labels_for_mapping(
    adata_base: ad.AnnData,
    cluster_key_base: str,
    adata_run: ad.AnnData,
    cluster_key_run: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align baseline/run by canonical barcodes and return matched label vectors.
    """
    base_bc = canon_barcodes(pd.Index(adata_base.obs_names))
    run_bc = canon_barcodes(pd.Index(adata_run.obs_names))

    common = sorted(set(base_bc).intersection(set(run_bc)))
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between baseline and run.")

    base_map = {k: i for i, k in enumerate(base_bc)}
    run_map = {k: i for i, k in enumerate(run_bc)}

    base_pos = [base_map[x] for x in common]
    run_pos = [run_map[x] for x in common]

    base_labels = adata_base.obs.iloc[base_pos][cluster_key_base].astype(str).to_numpy()
    run_labels = adata_run.obs.iloc[run_pos][cluster_key_run].astype(str).to_numpy()
    return base_labels, run_labels


def build_method_baselines(
    file_table: pd.DataFrame,
    methods: list[str],
    seed: int,
    cluster_key: Optional[str],
    baseline_tag: str = "amb0",
):
    """
    Load amb0 baseline for each method and create method-specific baseline palettes.
    """
    baselines = {}

    for method in methods:
        sub = file_table[
            (file_table["method"] == method) &
            (file_table["tag"] == baseline_tag) &
            (file_table["seed"] == seed)
        ]
        if sub.empty:
            continue

        fp = sub.iloc[0]["file"]
        adata = ad.read_h5ad(fp)
        ck = auto_pick_cluster_key(adata, cluster_key)
        base_labels = adata.obs[ck].astype(str).tolist()
        palette = build_baseline_palette(base_labels)

        baselines[method] = {
            "adata": adata,
            "cluster_key": ck,
            "palette": palette,
        }

    return baselines


def make_legend_from_palette(fig, palette: dict[str, tuple], max_cols: int = 6):
    import matplotlib.lines as mlines

    handles = [
        mlines.Line2D(
            [], [], linestyle="None", marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor=color, label=str(label)
        )
        for label, color in palette.items()
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(max_cols, len(handles)),
            frameon=False,
            fontsize=9,
        )


def plot_spatial_overlays(
    file_table: pd.DataFrame,
    out_dir: Path,
    methods: list[str],
    tags: list[str],
    seed: int,
    cluster_key: Optional[str] = None,
    baseline_tag: str = "amb0",
    ref_method: str = "RNA",
    spot_size: float = 16.0,
    prefix: str = "spatial_overlays",
):
    selected = file_table[
        file_table["method"].isin(methods) &
        file_table["tag"].isin(tags) &
        (file_table["seed"] == seed)
    ].copy()

    if selected.empty:
        raise ValueError("No files found for requested methods/tags/seed.")

    baselines = build_method_baselines(
        file_table=file_table,
        methods=methods,
        seed=seed,
        cluster_key=cluster_key,
        baseline_tag=baseline_tag,
    )
    if not baselines:
        raise ValueError("No amb0 baselines found for the requested methods and seed.")

    # One shared coordinate system for all methods
    ref_coords = build_reference_coords(
        file_table=file_table,
        seed=seed,
        ref_method=ref_method,
        ref_tag=baseline_tag,
    )

    print(
        "Reference coord range:",
        ref_coords["x"].min(), ref_coords["x"].max(),
        ref_coords["y"].min(), ref_coords["y"].max()
    )

    panel_data = {}

    for tag in tags:
        for method in methods:
            sub = selected[(selected["method"] == method) & (selected["tag"] == tag)]
            if sub.empty or method not in baselines:
                continue

            fp = sub.iloc[0]["file"]
            adata_run = ad.read_h5ad(fp)
            ck_run = auto_pick_cluster_key(adata_run, cluster_key)

            adata_base = baselines[method]["adata"]
            ck_base = baselines[method]["cluster_key"]
            palette = baselines[method]["palette"]

            # Match run clusters to that method's own baseline clusters
            base_labels_aligned, run_labels_aligned = align_labels_for_mapping(
                adata_base, ck_base, adata_run, ck_run
            )
            run_to_base = match_clusters_to_baseline(base_labels_aligned, run_labels_aligned)

            # Align labels to the shared reference coordinates
            coords_df, labels_ser = align_labels_to_reference(
                adata_run=adata_run,
                cluster_key=ck_run,
                ref_coords=ref_coords,
            )

            panel_data[(tag, method)] = {
                "coords": coords_df,
                "labels": labels_ser,
                "run_to_base": run_to_base,
                "palette": palette,
            }

            print(method, tag, len(coords_df), "aligned spots")

    if not panel_data:
        raise ValueError("No matching panels could be loaded.")

    fig, axes = plt.subplots(len(tags), len(methods), figsize=(4.2 * len(methods), 4.0 * len(tags)))
    axes = np.array(axes).reshape(len(tags), len(methods))

    legend_palette = None

    for i, tag in enumerate(tags):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            key = (tag, method)

            if key not in panel_data:
                ax.set_visible(False)
                continue

            panel = panel_data[key]
            coords_df = panel["coords"]
            labels_ser = panel["labels"]
            run_to_base = panel["run_to_base"]
            palette = panel["palette"]

            if legend_palette is None and method == methods[0]:
                legend_palette = palette

            x = coords_df["x"].to_numpy()
            y = coords_df["y"].to_numpy()

            colors = []
            for lab in labels_ser.tolist():
                if lab in run_to_base:
                    base_lab = run_to_base[lab]
                    colors.append(palette.get(base_lab, (0.7, 0.7, 0.7, 1.0)))
                else:
                    colors.append((0.7, 0.7, 0.7, 1.0))

            ax.scatter(x, y, c=colors, s=spot_size, linewidths=0, alpha=0.9)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(method_display(method))
            if j == 0:
                ax.text(
                    -0.02, 0.5, tag,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    rotation=90, fontsize=12
                )

    if legend_palette is not None:
        make_legend_from_palette(fig, legend_palette, max_cols=6)

    fig.suptitle(f"Spatial overlays across degradation (seed {seed})", fontsize=16)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    png = out_dir / f"{prefix}_seed{seed}.png"
    pdf = out_dir / f"{prefix}_seed{seed}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("[SAVE]", png)
    print("[SAVE]", pdf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["RNA", "mid_ae","clip", "spagcn"])
    ap.add_argument("--tags", nargs="+", default=["amb0","amb0p2", "amb0p4","amb0p6", "amb0p8"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster_key", default=None)
    ap.add_argument("--baseline_tag", default="amb0")
    ap.add_argument("--spot_size", type=float, default=16.0)
    ap.add_argument("--ref_method", default="RNA")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_h5ads(args.input_dir, recursive=args.recursive)
    if files.empty:
        raise ValueError("No matching .h5ad files found. Check --input_dir and --recursive.")

    print(f"Found {len(files)} parsed files")
    print(files.groupby("method").size())

    plot_spatial_overlays(
        file_table=files,
        out_dir=out_dir,
        methods=args.methods,
        tags=args.tags,
        seed=args.seed,
        cluster_key=args.cluster_key,
        baseline_tag=args.baseline_tag,
        ref_method=args.ref_method,
        spot_size=args.spot_size,
        prefix="spatial_overlays",
    )

if __name__ == "__main__":
    main()