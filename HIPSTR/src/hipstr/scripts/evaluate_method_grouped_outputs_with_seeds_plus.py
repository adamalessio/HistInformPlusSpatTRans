#!/usr/bin/env python3
"""
evaluate_method_grouped_outputs_with_seeds_plus.py

Evaluate clustered outputs saved with names like:
  early_ae_amb0_seed0.clustered_7clusters.h5ad
  early_ae_amb0p2_seed0.clustered_7clusters.h5ad
  clip_amb0_seed10.clustered_7clusters.h5ad
  clip_amb0p4_seed9.clustered_7clusters.h5ad

For each file, computes:
- ARI / NMI vs that method's amb0 baseline FROM THE SAME SEED
- embedding kNN Jaccard vs that method's amb0 baseline FROM THE SAME SEED
- ARI / NMI vs ground truth
- entropy / gini / n_clusters for run labels
- contingency-based failure mode summaries vs baseline
- graph-connectivity Jaccard and fragmentation stats when .obsp["connectivities"] exists
- Moran's I / Geary's C on the first embedding component using spatial kNN weights
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def canon_barcode_index(names: pd.Index) -> pd.Index:
    s = names.astype(str).str.strip()
    s = s.str.replace(r"\.\d+$", "", regex=True)
    s = s.str.replace(r"^.*_([ACGT]+-[0-9]+)$", r"\1", regex=True)
    return pd.Index(s)


def first_index_map(idx: pd.Index) -> dict[str, int]:
    out = {}
    for i, x in enumerate(idx.astype(str)):
        if x not in out:
            out[x] = i
    return out


def align_three(adata_run: ad.AnnData, adata_base: ad.AnnData, adata_truth: ad.AnnData | None = None):
    run_idx = canon_barcode_index(pd.Index(adata_run.obs_names))
    base_idx = canon_barcode_index(pd.Index(adata_base.obs_names))

    common = set(run_idx).intersection(set(base_idx))
    truth_map = None
    if adata_truth is not None:
        truth_idx = canon_barcode_index(pd.Index(adata_truth.obs_names))
        common = common.intersection(set(truth_idx))
        truth_map = first_index_map(truth_idx)

    common = sorted(common)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes among inputs.")

    run_map = first_index_map(run_idx)
    base_map = first_index_map(base_idx)

    run_pos = [run_map[x] for x in common]
    base_pos = [base_map[x] for x in common]
    truth_pos = [truth_map[x] for x in common] if truth_map is not None else None
    return common, run_pos, base_pos, truth_pos


def auto_pick_cluster_key(adata: ad.AnnData, preferred: str | None = None) -> str:
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


def auto_pick_rep_key(adata: ad.AnnData, preferred: str | None = None) -> str:
    if preferred is not None and preferred in adata.obsm:
        return preferred

    candidates = [
        "X_fused_concat",
        "X_fused_mid",
        "X_fused_late",
        "X_fused_early",
        "X_fused_clip",
        "X_rna_pca",
        "X_pca",
        "X_radiomics",
        "X_radiomics_pca",
    ]
    for c in candidates:
        if c in adata.obsm:
            return c

    if len(adata.obsm.keys()) > 0:
        return list(adata.obsm.keys())[0]

    raise KeyError("Could not infer embedding key from .obsm.")


def get_coords(adata: ad.AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        X = np.asarray(adata.obsm["spatial"])
        if X.ndim == 2 and X.shape[1] >= 2:
            return X[:, :2].astype(float, copy=False)

    obs = adata.obs
    for a, b in [
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
        ("x", "y"),
        ("col", "row"),
        ("x_pixel", "y_pixel"),
    ]:
        if a in obs.columns and b in obs.columns:
            return obs[[a, b]].to_numpy(dtype=float, copy=True)
    return None


def safe_ari_nmi(y1, y2):
    s1 = pd.Series(y1).astype(str)
    s2 = pd.Series(y2).astype(str)
    mask = s1.notna() & s2.notna() & (s1 != "nan") & (s2 != "nan") & (s1 != "None") & (s2 != "None")
    if mask.sum() < 2:
        return np.nan, np.nan, int(mask.sum())
    a = adjusted_rand_score(s1[mask], s2[mask])
    n = normalized_mutual_info_score(s1[mask], s2[mask])
    return float(a), float(n), int(mask.sum())


def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[0]
    k_eff = min(int(k), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    inds = nn.kneighbors(X, return_distance=False)
    return inds[:, 1:]


def mean_knn_jaccard(X1: np.ndarray, X2: np.ndarray, k: int = 30) -> float:
    I1 = knn_indices(X1, k)
    I2 = knn_indices(X2, k)

    vals = []
    for a, b in zip(I1, I2):
        sa = set(map(int, a))
        sb = set(map(int, b))
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        vals.append(inter / union if union > 0 else np.nan)
    return float(np.nanmean(vals))


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
    return pd.crosstab(pd.Series(b_labels.astype(str), name="baseline"),
                       pd.Series(r_labels.astype(str), name="run"))


def failure_mode_summaries(ct: pd.DataFrame) -> dict:
    row_nonzero = (ct > 0).sum(axis=1).to_numpy()
    row_weights = ct.sum(axis=1).to_numpy()
    frag = float((row_nonzero * row_weights).sum() / (row_weights.sum() + 1e-12))

    col_nonzero = (ct > 0).sum(axis=0).to_numpy()
    col_weights = ct.sum(axis=0).to_numpy()
    merg = float((col_nonzero * col_weights).sum() / (col_weights.sum() + 1e-12))

    dom_frac = (ct.max(axis=1) / (ct.sum(axis=1) + 1e-12)).to_numpy()
    pers = float((dom_frac * row_weights).sum() / (row_weights.sum() + 1e-12))
    return {"fragmentation_index": frag, "merging_index": merg, "cluster_persistence": pers}


def mean_row_jaccard_from_connectivities(Cb, Cr, eps=1e-12) -> float:
    Ab = (Cb > 0).astype(np.uint8).tocsr()
    Ar = (Cr > 0).astype(np.uint8).tocsr()
    Ab.setdiag(0); Ab.eliminate_zeros()
    Ar.setdiag(0); Ar.eliminate_zeros()

    inter = Ab.multiply(Ar).sum(axis=1).A1
    deg_b = Ab.sum(axis=1).A1
    deg_r = Ar.sum(axis=1).A1
    union = deg_b + deg_r - inter
    return float(np.mean(inter / (union + eps)))


def fragmentation_from_adjacency(A: sparse.csr_matrix) -> dict:
    n = int(A.shape[0])
    if n == 0:
        return {"n_components": 0, "giant_frac": float("nan"), "isolated_frac": float("nan")}
    A = (A > 0).astype(np.uint8).tocsr()
    A = ((A + A.T) > 0).astype(np.uint8)
    n_comp, labels = csgraph.connected_components(A, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_comp)
    giant = sizes.max() / n if n > 0 else 0.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    isolated = float(np.mean(deg == 0))
    return {"n_components": int(n_comp), "giant_frac": float(giant), "isolated_frac": float(isolated)}


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


def parse_method_tag_seed(filename: str):
    name = Path(filename).name

    m = re.match(
        r"(.+?)_((?:amb|cap|drop)[0-9p]+(?:__(?:amb|cap|drop)[0-9p]+)?)_seed([0-9]+)\.clustered_([0-9]+)clusters\.h5ad$",
        name
    )

    if not m:
        raise ValueError(f"Could not parse method/tag/seed from filename: {name}")

    method = m.group(1)
    tag = m.group(2)
    seed = int(m.group(3))
    n_clusters = int(m.group(4))

    return method, tag, seed, n_clusters

def find_grouped_files(input_dir: str, recursive: bool = True):
    p = Path(input_dir)
    files = sorted(p.rglob("*.h5ad") if recursive else p.glob("*.h5ad"))
    rows = []
    for fp in files:
        if not fp.is_file():
            continue
        try:
            method, tag, seed, n_clusters = parse_method_tag_seed(fp.name)
            rows.append({
                "file": str(fp),
                "method": method,
                "tag": tag,
                "seed": seed,
                "target_clusters_from_name": n_clusters,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def subset_connectivities(adata: ad.AnnData, positions: list[int]):
    if "connectivities" not in adata.obsp:
        return None
    return adata.obsp["connectivities"][positions, :][:, positions].tocsr()


def evaluate_one(run_path: str, baseline_path: str, truth_h5ad: ad.AnnData | None, truth_label_key: str | None,
                 cluster_key: str | None, rep_key: str | None, baseline_cluster_key: str | None,
                 baseline_rep_key: str | None, knn_k: int, spatial_k: int):
    run_h5ad = ad.read_h5ad(run_path)
    base_h5ad = ad.read_h5ad(baseline_path)

    cluster_key_run = auto_pick_cluster_key(run_h5ad, cluster_key)
    rep_key_run = auto_pick_rep_key(run_h5ad, rep_key)
    cluster_key_base = auto_pick_cluster_key(base_h5ad, baseline_cluster_key)
    rep_key_base = auto_pick_rep_key(base_h5ad, baseline_rep_key)

    common, run_pos, base_pos, truth_pos = align_three(run_h5ad, base_h5ad, truth_h5ad)

    run_clusters = run_h5ad.obs.iloc[run_pos][cluster_key_run].to_numpy()
    base_clusters = base_h5ad.obs.iloc[base_pos][cluster_key_base].to_numpy()

    ari_base, nmi_base, n_for_clusters = safe_ari_nmi(run_clusters, base_clusters)

    truth_ari = np.nan
    truth_nmi = np.nan
    n_for_truth = 0
    if truth_h5ad is not None and truth_label_key is not None:
        truth_labels = truth_h5ad.obs.iloc[truth_pos][truth_label_key].to_numpy()
        truth_ari, truth_nmi, n_for_truth = safe_ari_nmi(run_clusters, truth_labels)

    X_run = np.asarray(run_h5ad.obsm[rep_key_run])[run_pos]
    X_base = np.asarray(base_h5ad.obsm[rep_key_base])[base_pos]
    emb_jacc = mean_knn_jaccard(X_run, X_base, k=knn_k)

    ct = contingency(base_clusters, run_clusters)
    fail = failure_mode_summaries(ct)

    C_run = subset_connectivities(run_h5ad, run_pos)
    C_base = subset_connectivities(base_h5ad, base_pos)

    conn_jacc = np.nan
    run_n_components = np.nan
    run_giant_frac = np.nan
    run_isolated_frac = np.nan
    base_n_components = np.nan
    base_giant_frac = np.nan
    base_isolated_frac = np.nan

    if C_run is not None:
        frag_run = fragmentation_from_adjacency(C_run)
        run_n_components = frag_run["n_components"]
        run_giant_frac = frag_run["giant_frac"]
        run_isolated_frac = frag_run["isolated_frac"]

    if C_base is not None:
        frag_base = fragmentation_from_adjacency(C_base)
        base_n_components = frag_base["n_components"]
        base_giant_frac = frag_base["giant_frac"]
        base_isolated_frac = frag_base["isolated_frac"]

    if C_run is not None and C_base is not None:
        conn_jacc = mean_row_jaccard_from_connectivities(C_base, C_run)

    run_coords = get_coords(run_h5ad)
    base_coords = get_coords(base_h5ad)

    run_moran = np.nan
    run_geary = np.nan
    base_moran = np.nan
    base_geary = np.nan

    if run_coords is not None and len(run_pos) > 1:
        W_run = spatial_weights_knn(np.asarray(run_coords)[run_pos], k=spatial_k)
        run_moran = moran_i(X_run[:, 0], W_run)
        run_geary = geary_c(X_run[:, 0], W_run)

    if base_coords is not None and len(base_pos) > 1:
        W_base = spatial_weights_knn(np.asarray(base_coords)[base_pos], k=spatial_k)
        base_moran = moran_i(X_base[:, 0], W_base)
        base_geary = geary_c(X_base[:, 0], W_base)

    method, tag, seed, target_n = parse_method_tag_seed(Path(run_path).name)

    return {
        "file": str(run_path),
        "method": method,
        "tag": tag,
        "seed": seed,
        "target_clusters_from_name": target_n,
        "baseline_file": str(baseline_path),
        "cluster_key": cluster_key_run,
        "rep_key": rep_key_run,
        "baseline_cluster_key": cluster_key_base,
        "baseline_rep_key": rep_key_base,
        "n_common": len(common),
        "n_for_cluster_metrics": n_for_clusters,
        "n_for_truth_metrics": n_for_truth,
        "n_clusters_run": int(pd.Series(run_clusters.astype(str)).nunique()),
        "entropy_run": entropy_from_labels(run_clusters),
        "gini_run": gini_from_labels(run_clusters),
        "ARI_vs_baseline": ari_base,
        "NMI_vs_baseline": nmi_base,
        "kNN_Jaccard_vs_baseline": emb_jacc,
        "connectivity_Jaccard_vs_baseline": conn_jacc,
        "fragmentation_index": fail["fragmentation_index"],
        "merging_index": fail["merging_index"],
        "cluster_persistence": fail["cluster_persistence"],
        "run_n_components": run_n_components,
        "run_giant_frac": run_giant_frac,
        "run_isolated_frac": run_isolated_frac,
        "baseline_n_components": base_n_components,
        "baseline_giant_frac": base_giant_frac,
        "baseline_isolated_frac": base_isolated_frac,
        "run_moran_i": run_moran,
        "run_geary_c": run_geary,
        "baseline_moran_i": base_moran,
        "baseline_geary_c": base_geary,
        "ARI_vs_truth": truth_ari,
        "NMI_vs_truth": truth_nmi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--truth_h5ad", default=None)
    ap.add_argument("--truth_label_key", default=None)
    ap.add_argument("--cluster_key", default=None)
    ap.add_argument("--rep_key", default=None)
    ap.add_argument("--baseline_cluster_key", default=None)
    ap.add_argument("--baseline_rep_key", default=None)
    ap.add_argument("--knn_k", type=int, default=30)
    ap.add_argument("--spatial_k", type=int, default=10)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    grouped = find_grouped_files(args.input_dir, recursive=args.recursive)
    if grouped.empty:
        raise ValueError("No matching clustered .h5ad files found with the expected naming pattern.")

    baseline_map = {}
    for _, row in grouped.iterrows():
        if row["tag"] == "amb0":
            baseline_map[(row["method"], int(row["seed"]))] = row["file"]

    truth = None
    if args.truth_h5ad is not None:
        truth = ad.read_h5ad(args.truth_h5ad)
        if args.truth_label_key is None:
            raise ValueError("--truth_label_key is required when --truth_h5ad is provided")
        if args.truth_label_key not in truth.obs.columns:
            raise KeyError(f"{args.truth_label_key} not found in truth_h5ad .obs")

    rows = []
    for _, row in grouped.iterrows():
        method = row["method"]
        seed = int(row["seed"])
        run_file = row["file"]
        baseline_key = (method, seed)

        if baseline_key not in baseline_map:
            rows.append({
                "file": run_file,
                "method": method,
                "tag": row["tag"],
                "seed": seed,
                "error": f"No amb0 baseline found for method={method}, seed={seed}",
            })
            print(f"[FAIL] {run_file}: no amb0 baseline for method={method}, seed={seed}")
            continue

        try:
            out = evaluate_one(
                run_path=run_file,
                baseline_path=baseline_map[baseline_key],
                truth_h5ad=truth,
                truth_label_key=args.truth_label_key,
                cluster_key=args.cluster_key,
                rep_key=args.rep_key,
                baseline_cluster_key=args.baseline_cluster_key,
                baseline_rep_key=args.baseline_rep_key,
                knn_k=args.knn_k,
                spatial_k=args.spatial_k,
            )
            rows.append(out)
            print(f"[OK] {run_file}")
        except Exception as e:
            rows.append({
                "file": run_file,
                "method": method,
                "tag": row["tag"],
                "seed": seed,
                "error": str(e),
            })
            print(f"[FAIL] {run_file}: {e}")

    df = pd.DataFrame(rows).sort_values(["method", "seed", "tag"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
