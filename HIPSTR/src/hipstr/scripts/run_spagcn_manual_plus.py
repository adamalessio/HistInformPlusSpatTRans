#!/usr/bin/env python3
"""
run_spagcn_manual_plus.py

Run SpaGCN on a manually loaded degraded Visium-style dataset and compute
comparison metrics, including additional cluster-failure and spatial-structure
summaries.

Adds:
- entropy_run
- gini_run
- fragmentation_index
- merging_index
- cluster_persistence
- connectivity_Jaccard_vs_baseline (if baseline embedding csv exists)
- run_moran_i / run_geary_c on PCA-1
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
import torch
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import tifffile
import sys

print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHON VERSION:", sys.version)


def open_text_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def read_table_auto(path: str, sep: Optional[str] = None, header=None) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, sep=sep, header=header)
    with open_text_maybe_gzip(path) as fh:
        first = fh.readline()
    guessed_sep = "\t" if "\t" in first else ","
    return pd.read_csv(path, sep=guessed_sep, header=header)


def load_mtx_expression(
    matrix_path: str,
    barcodes_path: str,
    features_path: str,
    uppercase_genes: bool = True,
) -> AnnData:
    X = mmread(matrix_path)
    if not issparse(X):
        X = csr_matrix(X)
    else:
        X = X.tocsr()

    barcodes = read_table_auto(barcodes_path, sep="\t", header=None)
    features = read_table_auto(features_path, header=None)

    if X.shape[1] == len(barcodes) and X.shape[0] == len(features):
        X = X.T.tocsr()
    elif X.shape[0] == len(barcodes) and X.shape[1] == len(features):
        pass
    else:
        raise ValueError(
            f"Matrix shape {X.shape} does not match barcodes ({len(barcodes)}) "
            f"and features ({len(features)})."
        )

    obs_names = barcodes.iloc[:, 0].astype(str).values
    if features.shape[1] >= 2:
        var_names = features.iloc[:, 1].astype(str).values
        gene_ids = features.iloc[:, 0].astype(str).values
    else:
        var_names = features.iloc[:, 0].astype(str).values
        gene_ids = features.iloc[:, 0].astype(str).values

    if uppercase_genes:
        var_names = np.array([g.upper() for g in var_names], dtype=object)

    adata = AnnData(X)
    adata.obs_names = obs_names
    adata.var_names = pd.Index(var_names)
    adata.var["gene_id"] = gene_ids
    adata.var["genename"] = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def attach_positions(
    adata: AnnData,
    positions_path: str,
    barcode_col: int | str = 0,
    in_tissue_col: int | str = 1,
    array_row_col: int | str = 2,
    array_col_col: int | str = 3,
    pixel_row_col: int | str = 4,
    pixel_col_col: int | str = 5,
    sep: Optional[str] = None,
    header=None,
    filter_in_tissue: bool = True,
) -> AnnData:
    pos = read_table_auto(positions_path, sep=sep, header=header).copy()

    def get_col(df: pd.DataFrame, col):
        return df[col] if isinstance(col, str) else df.iloc[:, col]

    pos["barcode"] = get_col(pos, barcode_col).astype(str)
    pos["in_tissue"] = pd.to_numeric(get_col(pos, in_tissue_col), errors="coerce")
    pos["array_row"] = pd.to_numeric(get_col(pos, array_row_col), errors="coerce")
    pos["array_col"] = pd.to_numeric(get_col(pos, array_col_col), errors="coerce")
    pos["pixel_row"] = pd.to_numeric(get_col(pos, pixel_row_col), errors="coerce")
    pos["pixel_col"] = pd.to_numeric(get_col(pos, pixel_col_col), errors="coerce")
    pos = pos.set_index("barcode")

    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between expression matrix and positions file.")

    adata = adata[common].copy()
    pos = pos.loc[common].copy()

    adata.obs["in_tissue"] = pos["in_tissue"].values
    adata.obs["x_array"] = pos["array_row"].values
    adata.obs["y_array"] = pos["array_col"].values
    adata.obs["x_pixel"] = pos["pixel_row"].values
    adata.obs["y_pixel"] = pos["pixel_col"].values
    adata.obsm["spatial"] = np.c_[adata.obs["x_pixel"].to_numpy(), adata.obs["y_pixel"].to_numpy()]

    if filter_in_tissue:
        keep = adata.obs["in_tissue"].astype(float) == 1
        adata = adata[keep].copy()

    return adata


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_knn_jaccard(emb_run: np.ndarray, emb_base: np.ndarray, k: int = 15) -> float:
    if emb_run.shape[0] != emb_base.shape[0]:
        raise ValueError("Baseline and run embeddings must have the same number of observations.")
    n = emb_run.shape[0]
    k_eff = min(k + 1, n)

    nn_run = NearestNeighbors(n_neighbors=k_eff).fit(emb_run)
    nn_base = NearestNeighbors(n_neighbors=k_eff).fit(emb_base)

    idx_run = nn_run.kneighbors(return_distance=False)
    idx_base = nn_base.kneighbors(return_distance=False)

    jaccards = []
    for i in range(n):
        s1 = set(idx_run[i].tolist())
        s2 = set(idx_base[i].tolist())
        s1.discard(i)
        s2.discard(i)
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        jaccards.append(inter / union if union else 1.0)
    return float(np.mean(jaccards))


def safe_silhouette(emb: np.ndarray, labels: Iterable) -> float:
    labels = pd.Series(labels).astype(str).values
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(silhouette_score(emb, labels))
    except Exception:
        return np.nan


def load_label_table(path: str, barcode_col: str, label_col: str) -> pd.Series:
    df = pd.read_csv(path)
    s = df.set_index(barcode_col)[label_col]
    s.index = s.index.astype(str)
    return s


def load_label_baseline(path: str, barcode_col: str, label_col: str) -> pd.Series:
    df = pd.read_csv(path)
    s = df.set_index(barcode_col)[label_col]
    s.index = s.index.astype(str)
    return s


def entropy_from_labels(labels: np.ndarray) -> float:
    _, counts = np.unique(labels.astype(str), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def gini_from_labels(labels: np.ndarray) -> float:
    _, counts = np.unique(labels.astype(str), return_counts=True)
    counts = np.sort(counts.astype(float))
    n = len(counts)
    if n == 0:
        return 0.0
    cum = np.cumsum(counts)
    g = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(g)


def contingency(b_labels: np.ndarray, r_labels: np.ndarray) -> pd.DataFrame:
    return pd.crosstab(
        pd.Series(b_labels.astype(str), name="baseline"),
        pd.Series(r_labels.astype(str), name="run")
    )


def failure_mode_summaries(ct: pd.DataFrame) -> dict:
    row_nonzero = (ct > 0).sum(axis=1).to_numpy()
    row_weights = ct.sum(axis=1).to_numpy()
    frag = float((row_nonzero * row_weights).sum() / (row_weights.sum() + 1e-12))

    col_nonzero = (ct > 0).sum(axis=0).to_numpy()
    col_weights = ct.sum(axis=0).to_numpy()
    merg = float((col_nonzero * col_weights).sum() / (col_weights.sum() + 1e-12))

    dom_frac = (ct.max(axis=1) / (ct.sum(axis=1) + 1e-12)).to_numpy()
    pers = float((dom_frac * row_weights).sum() / (row_weights.sum() + 1e-12))
    return {
        "fragmentation_index": frag,
        "merging_index": merg,
        "cluster_persistence": pers,
    }


def spatial_weights_knn(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    n = int(coords.shape[0])
    if n == 0:
        return sparse.csr_matrix((0, 0), dtype=float)
    k_eff = int(min(k, max(n - 1, 0)))
    if k_eff <= 0:
        return sparse.csr_matrix((n, n), dtype=float)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="ball_tree").fit(coords)
    dist, ind = nn.kneighbors(coords, return_distance=True)
    dist = dist[:, 1:]
    ind = ind[:, 1:]

    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = ind.reshape(-1).astype(np.int64, copy=False)
    w = (1.0 / (dist.reshape(-1) + 1e-9)).astype(float, copy=False)

    W = sparse.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=float)
    W = W.maximum(W.T).tocsr()

    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sparse.diags(1.0 / rs) @ W


def moran_i(x: np.ndarray, W: sparse.csr_matrix) -> float:
    x = x.astype(float)
    n = x.shape[0]
    x = x - x.mean()
    num = float(x @ (W @ x))
    den = float((x * x).sum()) + 1e-12
    return float((n / (W.sum() + 1e-12)) * (num / den))


def geary_c(x: np.ndarray, W: sparse.csr_matrix) -> float:
    x = x.astype(float)
    n = x.shape[0]
    x = x - x.mean()
    Wcoo = W.tocoo()
    diff2 = (x[Wcoo.row] - x[Wcoo.col]) ** 2
    num = float((Wcoo.data * diff2).sum())
    den = float((x * x).sum()) + 1e-12
    return float(((n - 1) / (2 * (W.sum() + 1e-12))) * (num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--barcodes", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-tag", required=True)

    ap.add_argument("--positions-sep", default=None)
    ap.add_argument("--positions-header", default="none", choices=["none", "infer"])
    ap.add_argument("--barcode-col", default="0")
    ap.add_argument("--in-tissue-col", default="1")
    ap.add_argument("--array-row-col", default="2")
    ap.add_argument("--array-col-col", default="3")
    ap.add_argument("--pixel-row-col", default="4")
    ap.add_argument("--pixel-col-col", default="5")

    ap.add_argument("--histology", action="store_true", default=True)
    ap.add_argument("--no-histology", dest="histology", action="store_false")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=int, default=49)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--search-l-start", type=float, default=0.01)
    ap.add_argument("--search-l-end", type=float, default=1000.0)
    ap.add_argument("--search-l-tol", type=float, default=0.01)
    ap.add_argument("--search-l-max-run", type=int, default=100)
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--res", type=float, default=None)
    ap.add_argument("--res-start", type=float, default=0.2)
    ap.add_argument("--res-step", type=float, default=0.1)
    ap.add_argument("--res-tol", type=float, default=5e-3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--shape", default="hexagon", choices=["hexagon", "square"])
    ap.add_argument("--seed", type=int, default=100)

    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--spatial-k", type=int, default=10)
    ap.add_argument("--baseline-pred", default=None)
    ap.add_argument("--baseline-barcode-col", default="barcode")
    ap.add_argument("--baseline-label-col", default="refined_pred")
    ap.add_argument("--pathologist-labels", default=None)
    ap.add_argument("--pathologist-barcode-col", default="barcode")
    ap.add_argument("--pathologist-label-col", default="label")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    header = None if args.positions_header == "none" else "infer"

    def parse_col(v: str):
        return int(v) if v.isdigit() else v

    adata = load_mtx_expression(
        matrix_path=args.matrix,
        barcodes_path=args.barcodes,
        features_path=args.features,
        uppercase_genes=True,
    )
    adata = attach_positions(
        adata,
        positions_path=args.positions,
        barcode_col=parse_col(args.barcode_col),
        in_tissue_col=parse_col(args.in_tissue_col),
        array_row_col=parse_col(args.array_row_col),
        array_col_col=parse_col(args.array_col_col),
        pixel_row_col=parse_col(args.pixel_row_col),
        pixel_col_col=parse_col(args.pixel_col_col),
        sep=args.positions_sep,
        header=header,
        filter_in_tissue=True,
    )

    adata.raw = adata.copy()

    img = tifffile.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")

    x_array = adata.obs["x_array"].astype(int).tolist()
    y_array = adata.obs["y_array"].astype(int).tolist()
    x_pixel = adata.obs["x_pixel"].round().astype(int).tolist()
    y_pixel = adata.obs["y_pixel"].round().astype(int).tolist()

    if args.histology:
        adj = spg.calculate_adj_matrix(
            x=x_pixel,
            y=y_pixel,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            image=img,
            beta=args.beta,
            alpha=args.alpha,
            histology=True,
        )
    else:
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)

    np.savetxt(outdir / "adj.csv", adj, delimiter=",")

    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)

    if issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    l_val = spg.search_l(
        args.p,
        adj,
        start=args.search_l_start,
        end=args.search_l_end,
        tol=args.search_l_tol,
        max_run=args.search_l_max_run,
    )

    if args.res is not None:
        res = args.res
    elif args.n_clusters is not None:
        res = spg.search_res(
            adata,
            adj,
            l_val,
            args.n_clusters,
            start=args.res_start,
            step=args.res_step,
            tol=args.res_tol,
            lr=args.lr,
            max_epochs=args.max_epochs,
            r_seed=args.seed,
            t_seed=args.seed,
            n_seed=args.seed,
        )
    else:
        raise ValueError("Provide either --res or --n-clusters.")

    set_random_seeds(args.seed)
    clf = spg.SpaGCN()
    clf.set_l(l_val)
    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=args.res_tol,
        lr=args.lr,
        max_epochs=args.max_epochs,
    )
    y_pred, prob = clf.predict()
    adata.obs["pred"] = pd.Categorical(y_pred.astype(str))

    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = spg.refine(
        sample_id=adata.obs_names.tolist(),
        pred=adata.obs["pred"].astype(str).tolist(),
        dis=adj_2d,
        shape=args.shape,
    )
    adata.obs["refined_pred"] = pd.Categorical(pd.Series(refined_pred).astype(str).values)

    sc.pp.pca(adata, n_comps=min(50, max(2, min(adata.n_obs - 1, adata.n_vars - 1))))
    emb = np.asarray(adata.obsm["X_pca"])

    metrics = {
        "tag": args.sample_tag,
        "method": "banksy" if "banksy" in str(outdir).lower() else "spagcn",
        "seed": args.seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "p": args.p,
        "l": float(l_val),
        "resolution": float(res),
        "n_clusters_run": int(pd.Series(adata.obs["refined_pred"]).nunique()),
        "silhouette_run": safe_silhouette(emb, adata.obs["refined_pred"]),
        "entropy_run": entropy_from_labels(adata.obs["refined_pred"].astype(str).to_numpy()),
        "gini_run": gini_from_labels(adata.obs["refined_pred"].astype(str).to_numpy()),
        "ARI_vs_baseline": np.nan,
        "NMI_vs_baseline": np.nan,
        "kNN_Jaccard": np.nan,
        "fragmentation_index": np.nan,
        "merging_index": np.nan,
        "cluster_persistence": np.nan,
        "run_moran_i": np.nan,
        "run_geary_c": np.nan,
        "ARI_vs_pathologist": np.nan,
        "NMI_vs_pathologist": np.nan,
    }

    # run spatial autocorrelation on PCA1
    if emb.shape[0] > 1 and "spatial" in adata.obsm:
        W = spatial_weights_knn(np.asarray(adata.obsm["spatial"]), k=args.spatial_k)
        metrics["run_moran_i"] = moran_i(emb[:, 0], W)
        metrics["run_geary_c"] = geary_c(emb[:, 0], W)

    if args.baseline_pred is not None:
        base_labels = load_label_baseline(
            args.baseline_pred,
            barcode_col=args.baseline_barcode_col,
            label_col=args.baseline_label_col,
        )
        common = adata.obs_names.intersection(base_labels.index)
        if len(common) > 0:
            run_lab = adata.obs.loc[common, "refined_pred"].astype(str).values
            base_lab = base_labels.loc[common].astype(str).values
            metrics["ARI_vs_baseline"] = float(adjusted_rand_score(base_lab, run_lab))
            metrics["NMI_vs_baseline"] = float(normalized_mutual_info_score(base_lab, run_lab))

            ct = contingency(base_lab, run_lab)
            fail = failure_mode_summaries(ct)
            metrics["fragmentation_index"] = fail["fragmentation_index"]
            metrics["merging_index"] = fail["merging_index"]
            metrics["cluster_persistence"] = fail["cluster_persistence"]

            emb_csv = Path(args.baseline_pred).with_name(Path(args.baseline_pred).stem + "_embedding.csv")
            if emb_csv.exists():
                base_emb_df = pd.read_csv(emb_csv).set_index(args.baseline_barcode_col)
                emb_cols = [c for c in base_emb_df.columns if c != args.baseline_label_col]
                base_emb = base_emb_df.loc[common, emb_cols].to_numpy()
                run_emb = emb[[adata.obs_names.get_loc(b) for b in common], :]
                metrics["kNN_Jaccard"] = compute_knn_jaccard(run_emb, base_emb, k=args.knn_k)

    if args.pathologist_labels is not None:
        gt = load_label_table(
            args.pathologist_labels,
            barcode_col=args.pathologist_barcode_col,
            label_col=args.pathologist_label_col,
        )
        common = adata.obs_names.intersection(gt.index)
        if len(common) > 0:
            run_lab = adata.obs.loc[common, "refined_pred"].astype(str).values
            gt_lab = gt.loc[common].astype(str).values
            metrics["ARI_vs_pathologist"] = float(adjusted_rand_score(gt_lab, run_lab))
            metrics["NMI_vs_pathologist"] = float(normalized_mutual_info_score(gt_lab, run_lab))

    adata.write_h5ad(outdir / f"{args.sample_tag}.spagcn.h5ad")
    pd.DataFrame({
        "barcode": adata.obs_names,
        "pred": adata.obs["pred"].astype(str),
        "refined_pred": adata.obs["refined_pred"].astype(str),
        "X_pca_1": emb[:, 0] if emb.shape[1] > 0 else np.nan,
        "X_pca_2": emb[:, 1] if emb.shape[1] > 1 else np.nan,
    }).to_csv(outdir / f"{args.sample_tag}.spagcn_clusters.csv", index=False)
    pd.DataFrame([metrics]).to_csv(outdir / f"{args.sample_tag}.spagcn_metrics.csv", index=False)

    with open(outdir / f"{args.sample_tag}.spagcn_run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
