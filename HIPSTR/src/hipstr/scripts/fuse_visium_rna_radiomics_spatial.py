#!/usr/bin/env python3
"""
fuse_visium_rna_radiomics_spatial.py

Loads CellRanger filtered_feature_bc_matrix.h5 (10x HDF5),
attaches Visium/VisiumHD spatial metadata from spatial/ directory,
joins radiomics by barcode, saves a combined h5ad,
then trains and saves embeddings for:
  - Early fusion autoencoder
  - Mid fusion autoencoder
  - Late fusion autoencoder
  - CLIP-like contrastive dual-encoder

Outputs:
  out_prefix.combined.h5ad
  out_prefix.early_ae.h5ad
  out_prefix.mid_ae.h5ad
  out_prefix.late_ae.h5ad
  out_prefix.clip.h5ad
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image


# -----------------------------
# Utilities
# -----------------------------


def export_baseline_inputs(
    adata,
    out_dir: str,
    spatial_key_candidates=("spatial", "X_spatial", "spatial_coords"),
    make_spatial_graph: bool = True,
    spatial_k: int = 6,
) -> None:
    """Export minimal, method-agnostic inputs for baseline methods (BANKSY, BayesSpace, HMRF, SpiceMix, MERINGUE).

    Outputs (gzipped):
      - counts.mtx.gz  : raw (integer) counts matrix, shape (N, p)
      - genes.tsv.gz   : gene names (var_names)
      - barcodes.tsv.gz: spot/barcode ids (obs_names)
      - obs.csv.gz     : obs metadata (adata.obs)
      - spatial.csv.gz : 2D spatial coordinates
      - spatial_knn_edges.csv.gz (optional): undirected kNN graph on spatial coords
    """
    import os, gzip
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.io import mmwrite

    os.makedirs(out_dir, exist_ok=True)

    # --- counts ---
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)
    # Some pipelines store normalized floats in .X; try to prefer raw counts if present.
    if hasattr(adata, "layers") and ("counts" in adata.layers):
        X_counts = adata.layers["counts"]
        X_counts = X_counts.tocsr() if sp.issparse(X_counts) else np.asarray(X_counts)
        X = X_counts

    # Ensure integer-ish for interoperability (R packages often assume integer counts).
    if sp.issparse(X):
        X_int = X.copy()
        # If float but near-integers, round; otherwise keep as is.
        if X_int.dtype.kind in "fc":
            X_int.data = np.rint(X_int.data).astype(np.int64, copy=False)
        else:
            X_int.data = X_int.data.astype(np.int64, copy=False)
    else:
        if X.dtype.kind in "fc":
            X_int = np.rint(X).astype(np.int64)
        else:
            X_int = X.astype(np.int64)

    counts_path = os.path.join(out_dir, "counts.mtx")
    mmwrite(counts_path, X_int)
    with open(counts_path, "rb") as f_in, gzip.open(counts_path + ".gz", "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(counts_path)

    # --- genes / barcodes ---
    genes = pd.DataFrame({"gene": adata.var_names.astype(str)})
    barcodes = pd.DataFrame({"barcode": adata.obs_names.astype(str)})

    for df, fname in [(genes, "genes.tsv.gz"), (barcodes, "barcodes.tsv.gz")]:
        with gzip.open(os.path.join(out_dir, fname), "wt") as f:
            df.to_csv(f, sep="\t", index=False, header=False)

    # --- obs metadata ---
    obs_df = adata.obs.copy()
    obs_df.insert(0, "barcode", adata.obs_names.astype(str))
    with gzip.open(os.path.join(out_dir, "obs.csv.gz"), "wt") as f:
        obs_df.to_csv(f, index=False)

    # --- spatial coordinates ---
    spatial = None
    for k in spatial_key_candidates:
        if k in getattr(adata, "obsm", {}):
            spatial = adata.obsm[k]
            break
    if spatial is None:
        raise ValueError(
            "No spatial coordinates found in adata.obsm. Tried keys: "
            + ", ".join(spatial_key_candidates)
        )
    spatial = np.asarray(spatial)
    if spatial.shape[1] < 2:
        raise ValueError(f"Spatial coords must have at least 2 columns; got shape {spatial.shape}")

    spatial_df = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str),
            "x": spatial[:, 0],
            "y": spatial[:, 1],
        }
    )
    with gzip.open(os.path.join(out_dir, "spatial.csv.gz"), "wt") as f:
        spatial_df.to_csv(f, index=False)

    # --- optional: a simple spatial kNN graph (useful for HMRF / MRF baselines) ---
    if make_spatial_graph:
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(spatial_k + 1, spatial.shape[0]), metric="euclidean")
            nn.fit(spatial[:, :2])
            dists, inds = nn.kneighbors(spatial[:, :2], return_distance=True)

            # Build undirected edges i<j
            edges = []
            for i in range(inds.shape[0]):
                for jpos in range(1, inds.shape[1]):  # skip self at position 0
                    j = int(inds[i, jpos])
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    edges.append((a, b, float(dists[i, jpos])))

            edges_df = pd.DataFrame(edges, columns=["i", "j", "dist"])
            edges_df.drop_duplicates(subset=["i", "j"], inplace=True)
            # Map to barcodes for convenience
            edges_df["barcode_i"] = adata.obs_names.values[edges_df["i"].values]
            edges_df["barcode_j"] = adata.obs_names.values[edges_df["j"].values]
            edges_df = edges_df[["barcode_i", "barcode_j", "dist"]]

            with gzip.open(os.path.join(out_dir, "spatial_knn_edges.csv.gz"), "wt") as f:
                edges_df.to_csv(f, index=False)
        except Exception as e:
            print(f"[WARN] Failed to export spatial kNN edges: {e}")


def export_10x_mtx_dir(
    adata: ad.AnnData,
    out_10x_dir: str,
) -> None:
    """Export a 10x-style MTX directory (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz).

    This is useful because many downstream scripts/packages expect 10x MTX layout.
    We export Gene Expression only, with:
      col0: gene_id (if available in adata.var['gene_ids'], else var_names)
      col1: gene_symbol (var_names)
      col2: feature_type ('Gene Expression')
    """
    import os, gzip
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.io import mmwrite

    os.makedirs(out_10x_dir, exist_ok=True)

    X = adata.X
    if hasattr(adata, "layers") and ("counts" in adata.layers):
        X = adata.layers["counts"]
    if sp.issparse(X):
        X = X.tocsr()
        if X.dtype.kind in "fc":
            X = X.copy()
            X.data = np.rint(X.data).astype(np.int64, copy=False)
        else:
            X.data = X.data.astype(np.int64, copy=False)
    else:
        X = np.asarray(X)
        if X.dtype.kind in "fc":
            X = np.rint(X).astype(np.int64)
        else:
            X = X.astype(np.int64)

    # 10x MTX expects shape (genes, cells)
    if sp.issparse(X):
        X_10x = X.T.tocoo()
    else:
        X_10x = sp.coo_matrix(X.T)

    mtx_path = os.path.join(out_10x_dir, "matrix.mtx")
    mmwrite(mtx_path, X_10x)
    with open(mtx_path, "rb") as f_in, gzip.open(mtx_path + ".gz", "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(mtx_path)

    # barcodes
    bc_path = os.path.join(out_10x_dir, "barcodes.tsv.gz")
    with gzip.open(bc_path, "wt") as f:
        for b in adata.obs_names.astype(str):
            f.write(f"{b}\n")

    # features
    gene_symbols = adata.var_names.astype(str).tolist()
    if "gene_ids" in adata.var.columns:
        gene_ids = adata.var["gene_ids"].astype(str).tolist()
    else:
        gene_ids = gene_symbols

    feat_df = pd.DataFrame(
        {
            0: gene_ids,
            1: gene_symbols,
            2: ["Gene Expression"] * len(gene_symbols),
        }
    )
    feat_path = os.path.join(out_10x_dir, "features.tsv.gz")
    with gzip.open(feat_path, "wt") as f:
        feat_df.to_csv(f, sep="\t", index=False, header=False)


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def read_radiomics_table(path: str, barcode_col: str = None, sep: str = None) -> pd.DataFrame:
    """
    Radiomics file where barcodes are either:
      - stored in a column (barcode_col is that column name), OR
      - stored as the first column / pandas index (barcode_col is None or 'index').

    Keeps only numeric columns.
    """
    if sep is None:
        sep = "," if path.lower().endswith(".csv") else "\t"

    # Read without dtype guessing issues
    df = pd.read_csv(path, sep=sep, low_memory=False)

    # Case 1: barcodes are the index/rownames (common for pandas to_csv(index=True))
    if barcode_col is None or str(barcode_col).lower() in ("index", "rownames", "rowname", "obs_names"):
        # If pandas wrote index, it often comes back as "Unnamed: 0"
        if "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
        else:
            # Otherwise assume first column is barcode-like
            df = df.set_index(df.columns[0])
    else:
        # Case 2: barcodes are in a named column
        if barcode_col not in df.columns:
            raise ValueError(
                f"barcode_col='{barcode_col}' not found in radiomics file columns: {df.columns.tolist()[:20]} ..."
            )
        df = df.set_index(barcode_col)

    df.index = df.index.astype(str)

    # Keep numeric radiomics only
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] == 0:
        raise ValueError(
            "No numeric radiomics features found. "
            "If your features are strings, coerce them to numeric before saving."
        )
    return num_df


def standardize_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X).astype(np.float32)

def _amb_tag(f: float) -> str:
    """
    Format ambient fraction for filenames, e.g. 0.3 -> 'amb0p30'
    """
    f = float(f)
    s = f"{f:.4f}".rstrip("0").rstrip(".")  # keep tidy
    s = s.replace(".", "p")
    return f"amb{s}"


def simulate_ambient_rna(
    adata: ad.AnnData,
    ambient_fraction: float = 0.3,
    random_state: int = 0,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Sparse-friendly ambient RNA simulation.

    Strategy:
      - Work in CSR sparse format.
      - Binomial thinning on existing counts (kept part).
      - For each cell i, sample n_ambient_i counts from a global ambient profile
        via Multinomial, build a sparse ambient matrix row-by-row.
      - Return a new AnnData with X as sparse CSR.
    """
    if ambient_fraction <= 0.0:
        if verbose:
            print("ambient_fraction <= 0.0, skipping degradation and returning copy of input AnnData.")
        return adata.copy()

    rng = np.random.default_rng(random_state)

    X = adata.X

    # Ensure CSR sparse
    if sparse.issparse(X):
        X_csr = X.tocsr()
    else:
        if verbose:
            print("Input X is dense; converting to CSR sparse...")
        X_csr = csr_matrix(X)

    n_cells, n_genes = X_csr.shape
    if verbose:
        print(f"Simulating ambient RNA on sparse matrix with shape {X_csr.shape} and nnz={X_csr.nnz}")

    # Library sizes per spot
    lib_sizes = np.asarray(X_csr.sum(axis=1)).ravel()

    # Global ambient profile: normalized gene totals
    gene_totals = np.asarray(X_csr.sum(axis=0)).ravel()
    gene_totals = gene_totals + 1e-8  # avoid division by zero
    ambient_profile = gene_totals / gene_totals.sum()

    # -------------------------
    # Part 1: Binomial thinning (kept counts)
    # -------------------------
    if verbose:
        print("  - Binomial thinning of existing counts (kept part)...")

    X_coo = X_csr.tocoo()
    rows = X_coo.row
    cols = X_coo.col
    data = X_coo.data.astype(np.int64)

    kept_data = rng.binomial(data, 1.0 - ambient_fraction).astype(np.int64)
    kept_mask = kept_data > 0

    kept = coo_matrix(
        (kept_data[kept_mask], (rows[kept_mask], cols[kept_mask])),
        shape=X_csr.shape
    ).tocsr()

    # -------------------------
    # Part 2: Ambient counts per cell
    # -------------------------
    if verbose:
        print("  - Sampling ambient counts per cell (sparse)...")

    amb_rows = []
    amb_cols = []
    amb_data = []

    for i in range(n_cells):
        n_ambient = int(round(ambient_fraction * lib_sizes[i]))
        if n_ambient <= 0:
            continue

        ambient_i = rng.multinomial(n_ambient, ambient_profile)
        nz = ambient_i.nonzero()[0]
        if nz.size == 0:
            continue

        amb_rows.append(np.full(nz.shape[0], i, dtype=np.int64))
        amb_cols.append(nz.astype(np.int64))
        amb_data.append(ambient_i[nz].astype(np.int64))

    if amb_data:
        amb_rows = np.concatenate(amb_rows)
        amb_cols = np.concatenate(amb_cols)
        amb_data = np.concatenate(amb_data)

        ambient_sparse = coo_matrix(
            (amb_data, (amb_rows, amb_cols)),
            shape=X_csr.shape
        ).tocsr()
    else:
        ambient_sparse = csr_matrix(X_csr.shape, dtype=np.int64)

    # -------------------------
    # Combine kept + ambient
    # -------------------------
    X_degraded = (kept + ambient_sparse).astype(np.int64)

    if verbose:
        print("Ambient simulation complete.")
        print(f"  Original mean counts per spot: {lib_sizes.mean():.1f}")
        print(f"  Degraded mean counts per spot: {np.asarray(X_degraded.sum(axis=1)).ravel().mean():.1f}")
        print(f"  Degraded nnz: {X_degraded.nnz}")

    # Build new AnnData; copy obs/var and metadata
    adata_deg = ad.AnnData(
        X=X_degraded,
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )

    # copy obsm/uns/layers so spatial is preserved
    adata_deg.obsm = adata.obsm.copy()
    adata_deg.uns = adata.uns.copy()
    if hasattr(adata, "layers"):
        adata_deg.layers = adata.layers.copy()

    return adata_deg


def _drop_tag(p: float) -> str:
    p = float(p)
    s = f"{p:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"drop{s}"


def _cap_tag(mean_keep: float) -> str:
    mean_keep = float(mean_keep)
    s = f"{mean_keep:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"cap{s}"


def simulate_sparse_entry_dropout(
    adata: ad.AnnData,
    dropout_prob: float,
    random_state: int = 0,
    verbose: bool = True,
) -> ad.AnnData:
    """Second degradation model (dropout): randomly delete a fraction of nonzero entries in the count matrix.

    This increases sparsity by removing observed nonzero molecules, distinct from ambient contamination.

    Works on sparse CSR efficiently.
    """
    from scipy import sparse as _sparse
    from scipy.sparse import csr_matrix as _csr_matrix
    import numpy as _np

    p = float(dropout_prob)
    if p <= 0.0:
        if verbose:
            print("[drop] dropout_prob <= 0.0, skipping.")
        return adata.copy()
    if p >= 1.0:
        if verbose:
            print("[drop] dropout_prob >= 1.0, zeroing all counts.")
        out = adata.copy()
        out.X = _csr_matrix(out.X.shape, dtype=_np.int64)
        return out

    rng = _np.random.default_rng(int(random_state))

    X = adata.layers["counts"] if ("counts" in getattr(adata, "layers", {})) else adata.X
    if _sparse.issparse(X):
        X = X.tocsr()
    else:
        if verbose:
            print("[drop] Input X is dense; converting to CSR sparse...")
        X = _csr_matrix(_np.asarray(X))

    indptr = X.indptr
    indices = X.indices
    data = X.data

    new_indptr = _np.zeros_like(indptr)
    new_indices_chunks = []
    new_data_chunks = []

    nnz_accum = 0
    for i in range(X.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if b > a:
            keep = rng.random(b - a) >= p
            idx_keep = indices[a:b][keep]
            dat_keep = data[a:b][keep]
            if dat_keep.size > 0:
                new_indices_chunks.append(idx_keep)
                new_data_chunks.append(dat_keep)
                nnz_accum += dat_keep.size
        new_indptr[i + 1] = nnz_accum

    new_indices = _np.concatenate(new_indices_chunks) if new_indices_chunks else _np.array([], dtype=indices.dtype)
    new_data = _np.concatenate(new_data_chunks) if new_data_chunks else _np.array([], dtype=data.dtype)

    X2 = _csr_matrix((new_data, new_indices, new_indptr), shape=X.shape)

    out = adata.copy()
    out.X = X2
    return out


def simulate_capture_efficiency_thinning(
    adata: ad.AnnData,
    mean_keep: float = 0.7,
    sigma: float = 0.15,
    random_state: int = 0,
    verbose: bool = True,
) -> ad.AnnData:
    """Second degradation model (capture efficiency): per-spot binomial thinning with per-spot keep rates.

    keep_i ~ LogNormal(log(mean_keep), sigma), clipped to [0, 1]
    x_ij ~ Binomial(x_ij, keep_i)

    This models variable capture yield / measurement fidelity distinct from ambient contamination.
    """
    import numpy as _np
    from scipy import sparse as _sparse
    from scipy.sparse import csr_matrix as _csr_matrix

    mk = float(mean_keep)
    sig = float(sigma)
    rng = _np.random.default_rng(int(random_state))

    X = adata.layers["counts"] if ("counts" in getattr(adata, "layers", {})) else adata.X
    if _sparse.issparse(X):
        X = X.tocsr()
        data = X.data.copy()
        indptr = X.indptr
        indices = X.indices
    else:
        X = _np.asarray(X)

    n = X.shape[0]
    keep = rng.lognormal(mean=_np.log(max(mk, 1e-6)), sigma=max(sig, 0.0), size=n)
    keep = _np.clip(keep, 0.0, 1.0).astype(_np.float32)

    if _sparse.issparse(X):
        for i in range(n):
            a, b = indptr[i], indptr[i + 1]
            if b > a:
                p = float(keep[i])
                data[a:b] = rng.binomial(data[a:b].astype(_np.int64, copy=False), p).astype(data.dtype, copy=False)
        X2 = _csr_matrix((data, indices, indptr), shape=X.shape)
        X2.eliminate_zeros()
    else:
        X2 = X.astype(_np.int64, copy=True)
        for i in range(n):
            X2[i, :] = rng.binomial(X2[i, :], float(keep[i]))
        X2 = _csr_matrix(X2)

    out = adata.copy()
    out.X = X2
    out.obs["capture_keep"] = keep
    return out

def compute_rna_pca(adata: ad.AnnData, n_hvg: int, n_pcs: int, seed: int = 0) -> np.ndarray:
    tmp = adata.copy()
    sc.pp.filter_genes(tmp, min_counts=1)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)

    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat_v3")
    tmp = tmp[:, tmp.var["highly_variable"]].copy()

    X = tmp.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    svd = TruncatedSVD(n_components=n_pcs, random_state=seed)
    X_pca = svd.fit_transform(X).astype(np.float32)
    return X_pca


def save_with_embedding(base_adata: ad.AnnData, embedding: np.ndarray, key: str, out_path: str):
    out = base_adata.copy()
    out.obsm[key] = embedding.astype(np.float32)
    out.write_h5ad(out_path)


# -----------------------------
# Attach Visium spatial info
# -----------------------------

def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def attach_visium_spatial(
    adata: ad.AnnData,
    spatial_dir: str,
    library_id: str = "visium",
    load_images: bool = True,
) -> ad.AnnData:
    """
    Attach CellRanger Visium spatial metadata to an AnnData created from read_10x_h5.

    Expected in spatial_dir:
      - tissue_positions.csv or tissue_positions_list.csv (or similar)
      - scalefactors_json.json
      - tissue_hires_image.png and/or tissue_lowres_image.png

    Produces a Scanpy-compatible structure for sc.pl.spatial:
      adata.obsm["spatial"] -> (n_spots, 2) fullres pixel coords [x, y]
      adata.uns["spatial"][library_id]["images"] -> dict with 'hires'/'lowres'
      adata.uns["spatial"][library_id]["scalefactors"] -> dict
      adata.obs includes in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    """
    if spatial_dir is None:
        return adata
    if not os.path.isdir(spatial_dir):
        raise ValueError(f"--spatial_dir does not exist or is not a directory: {spatial_dir}")

    # Tissue positions file names vary a bit across CellRanger versions
    pos_path = _find_first_existing([
        os.path.join(spatial_dir, "tissue_positions.csv"),
        os.path.join(spatial_dir, "tissue_positions_list.csv"),
        os.path.join(spatial_dir, "tissue_positions.parquet"),
    ])
    if pos_path is None:
        raise FileNotFoundError(
            f"Could not find tissue_positions*.csv in spatial_dir={spatial_dir}"
        )

    if pos_path.endswith(".parquet"):
        pos = pd.read_parquet(pos_path)
    else:
        # Handle both formats:
        # - tissue_positions_list.csv has no header in older versions (5 columns)
        # - tissue_positions.csv has header in newer versions (6 columns)
        pos_raw = pd.read_csv(pos_path, header=None, low_memory=False)
        if pos_raw.shape[1] >= 6:
            # likely no header but 6 cols: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col
            pos_raw.columns = [
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres"
            ]
        else:
            raise ValueError(f"Unexpected tissue positions format in {pos_path} with {pos_raw.shape[1]} columns.")
        pos = pos_raw

    pos["barcode"] = pos["barcode"].astype(str)
    pos = pos.set_index("barcode")

    # Align to adata spots (barcodes)
    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise ValueError("No overlap between adata.obs_names and tissue positions barcodes.")
    adata = adata[common].copy()
    pos = pos.loc[common]

    # store in obs
    for c in ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        if c in pos.columns:
            adata.obs[c] = pos[c].values

    # Scanpy expects adata.obsm["spatial"] as (x, y) pixel coords (col, row)
    if "pxl_col_in_fullres" in pos.columns and "pxl_row_in_fullres" in pos.columns:
        adata.obsm["spatial"] = np.vstack([
            pos["pxl_col_in_fullres"].values,
            pos["pxl_row_in_fullres"].values
        ]).T.astype(np.float32)

    # scalefactors
    sf_path = os.path.join(spatial_dir, "scalefactors_json.json")
    scalefactors = {}
    if os.path.exists(sf_path):
        with open(sf_path, "r") as f:
            scalefactors = json.load(f)

    # images
    images = {}
    if load_images:
        hires = os.path.join(spatial_dir, "tissue_hires_image.png")
        lowres = os.path.join(spatial_dir, "tissue_lowres_image.png")
        if os.path.exists(hires):
            images["hires"] = np.array(Image.open(hires))
        if os.path.exists(lowres):
            images["lowres"] = np.array(Image.open(lowres))

    # attach in standard Scanpy structure
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {
        "images": images,
        "scalefactors": scalefactors,
        "metadata": {"source": "cellranger", "spatial_dir": spatial_dir},
    }

    # helpful for plotting: tell scanpy which library to use
    adata.uns["spatial"][library_id]["use_quality"] = "hires" if "hires" in images else "lowres"

    return adata


# -----------------------------
# Torch datasets and models
# -----------------------------

class PairDataset(Dataset):
    def __init__(self, X_rna: np.ndarray, X_path: np.ndarray):
        assert X_rna.shape[0] == X_path.shape[0]
        self.X_rna = torch.from_numpy(X_rna)
        self.X_path = torch.from_numpy(X_path)

    def __len__(self):
        return self.X_rna.shape[0]

    def __getitem__(self, idx):
        return self.X_rna[idx], self.X_path[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EarlyFusionAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, z_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, in_dim)
        )

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return z, xhat


class MidFusionAE(nn.Module):
    def __init__(self, rna_dim: int, path_dim: int, hidden: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.r_enc = nn.Sequential(nn.Linear(rna_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.p_enc = nn.Sequential(nn.Linear(path_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.shared = nn.Sequential(nn.Linear(2 * hidden, z_dim))

        self.r_dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                    nn.Linear(hidden, rna_dim))
        self.p_dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                    nn.Linear(hidden, path_dim))

    def encode(self, rna, path):
        h = torch.cat([self.r_enc(rna), self.p_enc(path)], dim=1)
        return self.shared(h)

    def forward(self, rna, path):
        z = self.encode(rna, path)
        r_hat = self.r_dec(z)
        p_hat = self.p_dec(z)
        return z, r_hat, p_hat


class LateFusionAE(nn.Module):
    def __init__(self, rna_dim: int, path_dim: int, hidden: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.r_enc = nn.Sequential(nn.Linear(rna_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(hidden, z_dim))
        self.r_dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(hidden, rna_dim))

        self.p_enc = nn.Sequential(nn.Linear(path_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(hidden, z_dim))
        self.p_dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(hidden, path_dim))

        self.fuse = nn.Sequential(nn.Linear(2 * z_dim, z_dim))

    def encode(self, rna, path):
        z_r = self.r_enc(rna)
        z_p = self.p_enc(path)
        z = self.fuse(torch.cat([z_r, z_p], dim=1))
        return z, z_r, z_p

    def forward(self, rna, path):
        z, z_r, z_p = self.encode(rna, path)
        r_hat = self.r_dec(z_r)
        p_hat = self.p_dec(z_p)
        return z, r_hat, p_hat


class CLIPDualEncoder(nn.Module):
    def __init__(self, rna_dim: int, path_dim: int, hidden: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.r = MLP(rna_dim, hidden, z_dim, dropout=dropout)
        self.p = MLP(path_dim, hidden, z_dim, dropout=dropout)

    def encode(self, rna, path):
        zr = F.normalize(self.r(rna), dim=1)
        zp = F.normalize(self.p(path), dim=1)
        return zr, zp


def clip_loss(zr, zp, temperature: float = 0.07):
    logits = (zr @ zp.t()) / temperature
    labels = torch.arange(zr.size(0), device=zr.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# -----------------------------
# Training loops
# -----------------------------

def train_early_ae(X_concat, hidden, z_dim, epochs, batch_size, lr, wd, device):
    model = EarlyFusionAE(in_dim=X_concat.shape[1], hidden=hidden, z_dim=z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_concat))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            _, xhat = model(xb)
            loss = F.mse_loss(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Z = []
        dl2 = DataLoader(torch.from_numpy(X_concat), batch_size=batch_size, shuffle=False)
        for xb in dl2:
            xb = xb.to(device)
            Z.append(model.encode(xb).cpu().numpy())
    return np.vstack(Z).astype(np.float32)


def train_mid_ae(X_rna, X_path, hidden, z_dim, epochs, batch_size, lr, wd, device):
    model = MidFusionAE(X_rna.shape[1], X_path.shape[1], hidden, z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for _ in range(epochs):
        for rna, path in dl:
            rna, path = rna.to(device), path.to(device)
            _, r_hat, p_hat = model(rna, path)
            loss = F.mse_loss(r_hat, rna) + F.mse_loss(p_hat, path)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Z = []
        dl2 = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=False)
        for rna, path in dl2:
            rna, path = rna.to(device), path.to(device)
            Z.append(model.encode(rna, path).cpu().numpy())
    return np.vstack(Z).astype(np.float32)


def train_late_ae(X_rna, X_path, hidden, z_dim, epochs, batch_size, lr, wd, device):
    model = LateFusionAE(X_rna.shape[1], X_path.shape[1], hidden, z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for _ in range(epochs):
        for rna, path in dl:
            rna, path = rna.to(device), path.to(device)
            _, r_hat, p_hat = model(rna, path)
            loss = F.mse_loss(r_hat, rna) + F.mse_loss(p_hat, path)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Z = []
        dl2 = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=False)
        for rna, path in dl2:
            rna, path = rna.to(device), path.to(device)
            z, _, _ = model.encode(rna, path)
            Z.append(z.cpu().numpy())
    return np.vstack(Z).astype(np.float32)


def train_clip(X_rna, X_path, hidden, z_dim, epochs, batch_size, lr, wd, temperature, device):
    model = CLIPDualEncoder(X_rna.shape[1], X_path.shape[1], hidden, z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for _ in range(epochs):
        for rna, path in dl:
            rna, path = rna.to(device), path.to(device)
            zr, zp = model.encode(rna, path)
            loss = clip_loss(zr, zp, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Z = []
        dl2 = DataLoader(PairDataset(X_rna, X_path), batch_size=batch_size, shuffle=False)
        for rna, path in dl2:
            rna, path = rna.to(device), path.to(device)
            zr, zp = model.encode(rna, path)
            z = F.normalize((zr + zp) / 2.0, dim=1)
            Z.append(z.cpu().numpy())
    return np.vstack(Z).astype(np.float32)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_h5", required=True, help="CellRanger filtered_feature_bc_matrix.h5 (10x HDF5)")
    ap.add_argument("--spatial_dir", default=None,
                    help="Path to CellRanger 'spatial/' directory (for sc.pl.spatial)")
    ap.add_argument("--library_id", default="visium", help="Key used under adata.uns['spatial'][library_id]")
    ap.add_argument("--no_images", action="store_true", help="Do not load hires/lowres images")

    ap.add_argument("--radiomics", required=True, help="Radiomics table (CSV/TSV) with barcode column + numeric features")
    ap.add_argument("--barcode_col", default="index",
                help="Barcode column in radiomics file. Use 'index' if barcodes are stored as rownames/index.")
    ap.add_argument("--radiomics_sep", default=None)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", default="visiumhd_fusion")
    ap.add_argument(
        "--export_baselines", action="store_true",
        help="If set, export counts + spatial inputs for baseline methods (BANKSY/BayesSpace/HMRF/SpiceMix/MERINGUE) for each degradation tag."
    )
    ap.add_argument(
        "--baseline_dir", default="baselines",
        help="Subdirectory under out_dir where baseline input files will be written (one folder per degradation tag)."
    )
    ap.add_argument(
        "--baseline_spatial_k", type=int, default=6,
        help="k for spatial kNN graph export (edges file) when --export_baselines is set."
    )

    ap.add_argument(
        "--export_tenx_mtx", action="store_true",
        help="If set, export a 10x-style MTX directory (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz) "
             "for each degradation tag. This makes it easy to run downstream RNA-only methods that read 10x MTX."
    )
    ap.add_argument(
        "--tenx_dirname", default="filtered_feature_bc_matrix",
        help="Subdirectory name to use under each baseline tag when --export_tenx_mtx is set."
    )

    ap.add_argument("--n_hvg", type=int, default=3000)
    ap.add_argument("--rna_pcs", type=int, default=128)
    ap.add_argument("--radiomics_pcs", type=int, default=128)

    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
    "--ambient_fractions",
    default="0.0",
    help="Comma-separated ambient fractions to simulate (e.g., '0,0.1,0.3'). "
         "Each value f adds ~f*library_size ambient counts and thins existing counts by (1-f)."
    )
    ap.add_argument(
    "--ambient_seed_offset",
    type=int,
    default=0,
    help="Offset added to --seed for ambient RNG per fraction (deterministic)."
    )


    ap.add_argument(
        "--dropout_probs",
        default="",
        help="Comma-separated dropout probabilities applied by deleting a fraction of nonzero entries (e.g. '0.05,0.1'). "
             "Produces tags like drop0p05. If --apply_secondary_on_ambient is set, produces amb0p1__drop0p05."
    )
    ap.add_argument(
        "--capture_keep_fracs",
        default="",
        help="Comma-separated mean keep fractions for capture-efficiency thinning (e.g. '0.95,0.9,0.8,0.7'). "
             "Produces tags like cap0p9. If --apply_secondary_on_ambient is set, produces amb0p1__cap0p7."
    )
    ap.add_argument(
        "--capture_sigma",
        type=float,
        default=0.15,
        help="Lognormal sigma controlling per-spot variability for capture-efficiency thinning."
    )
    ap.add_argument(
        "--apply_secondary_on_ambient",
        action="store_true",
        help="If set, apply dropout/capture degradations on top of each ambient-degraded dataset (tag becomes ambX__dropY / ambX__capY). "
             "If not set, dropout/capture tags are generated from the original data (in addition to ambient tags)."
    )

    args = ap.parse_args()
    set_seed(args.seed)
    safe_makedirs(args.out_dir)

    # Load RNA
    adata = sc.read_10x_h5(args.filtered_h5)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)

    # Attach spatial metadata + images
    if args.spatial_dir is not None:
        adata = attach_visium_spatial(
            adata,
            spatial_dir=args.spatial_dir,
            library_id=args.library_id,
            load_images=(not args.no_images),
        )

    # NOTE: baseline-method exports (counts/spatial/10x MTX) are performed per ambient fraction tag inside the loop below.

    # Load radiomics and join by barcode
    rad = read_radiomics_table(args.radiomics, args.barcode_col, args.radiomics_sep)
    common = adata.obs_names.intersection(rad.index)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between RNA (adata.obs_names) and radiomics index.")
    adata = adata[common].copy()
    rad = rad.loc[common]

    X_rad = standardize_features(rad.to_numpy(dtype=np.float32))
    adata.obsm["X_radiomics_raw"] = rad.to_numpy(dtype=np.float32)
    adata.obsm["X_radiomics"] = X_rad
    adata.uns["radiomics_feature_names"] = rad.columns.to_list()

    # Parse ambient fractions
    amb_fracs = [float(s.strip()) for s in args.ambient_fractions.split(",") if s.strip() != ""]
    if len(amb_fracs) == 0:
        amb_fracs = [0.0]

    device = torch.device(args.device)
# --- Build run list (ambient + optional secondary degradations) ---
    drop_probs = [float(s.strip()) for s in args.dropout_probs.split(",") if s.strip() != ""]
    cap_keeps  = [float(s.strip()) for s in args.capture_keep_fracs.split(",") if s.strip() != ""]

    runs = []

# Ambient runs (existing behavior)
    for amb in amb_fracs:
        tag = _amb_tag(amb)
        ad_amb = simulate_ambient_rna(
            adata,
            ambient_fraction=amb,
            random_state=int(args.seed + args.ambient_seed_offset + round(amb * 10_000)),
            verbose=True,
        )
        runs.append((tag, ad_amb))

        if args.apply_secondary_on_ambient:
            for p in drop_probs:
                tag2 = f"{tag}__{_drop_tag(p)}"
                ad2 = simulate_sparse_entry_dropout(
                    ad_amb,
                    dropout_prob=p,
                    random_state=int(args.seed + 100_000 + round(p * 10_000) + round(amb * 10_000)),
                    verbose=True,
                )
                runs.append((tag2, ad2))

            for mk in cap_keeps:
                tag2 = f"{tag}__{_cap_tag(mk)}"
                ad2 = simulate_capture_efficiency_thinning(
                    ad_amb,
                    mean_keep=mk,
                    sigma=float(args.capture_sigma),
                    random_state=int(args.seed + 200_000 + round(mk * 10_000) + round(amb * 10_000)),
                    verbose=True,
                )
                runs.append((tag2, ad2))

# Secondary-only runs (generated from original) unless applying on ambient
    if not args.apply_secondary_on_ambient:
        for p in drop_probs:
            tag = _drop_tag(p)
            ad2 = simulate_sparse_entry_dropout(
                adata,
                dropout_prob=p,
                random_state=int(args.seed + 100_000 + round(p * 10_000)),
                verbose=True,
            )
            runs.append((tag, ad2))

        for mk in cap_keeps:
            tag = _cap_tag(mk)
            ad2 = simulate_capture_efficiency_thinning(
                adata,
                mean_keep=mk,
                sigma=float(args.capture_sigma),
                random_state=int(args.seed + 200_000 + round(mk * 10_000)),
                verbose=True,
            )
            runs.append((tag, ad2))

# --- Run pipeline for each degradation tag ---
    last_combined_path = None

    for tag, adata_run in runs:
        print("\n" + "=" * 80)
        print(f"[RUN] tag={tag}")
        print("=" * 80)

    # Export method-agnostic baseline inputs + (optionally) 10x MTX layout per tag
        if args.export_baselines:
            baseline_tag_dir = os.path.join(args.out_dir, args.baseline_dir, tag)
            export_baseline_inputs(
                adata_run,
                baseline_tag_dir,
                make_spatial_graph=True,
                spatial_k=args.baseline_spatial_k,
            )
        if args.export_tenx_mtx:
            tenx_out = os.path.join(args.out_dir, args.baseline_dir, tag, args.tenx_dirname)
            export_10x_mtx_dir(adata_run, tenx_out)

    # RNA PCA from degraded counts
        X_rna = standardize_features(
            compute_rna_pca(adata_run, args.n_hvg, args.rna_pcs, seed=args.seed)
        )
        adata_run.obsm["X_rna_pca"] = X_rna

    # Radiomics already in adata_run.obsm from the joined base 'adata'
        X_rad_std = adata_run.obsm["X_radiomics"].astype(np.float32)

    # Optional PCA compression of radiomics for models
        if X_rad_std.shape[1] > args.radiomics_pcs:
            svd = TruncatedSVD(n_components=args.radiomics_pcs, random_state=args.seed)
            X_path = standardize_features(svd.fit_transform(X_rad_std).astype(np.float32))
            adata_run.obsm["X_radiomics_pca"] = X_path
            path_for_models = X_path
        else:
            path_for_models = X_rad_std

    # -------------------------
    # Save combined + embeddings (PER TAG)
    # -------------------------
        combined_path = os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.combined.h5ad")
        adata_run.write_h5ad(combined_path)
        last_combined_path = combined_path

    # Concatenate standardized RNA + radiomics (or radiomics_pca)
        X_concat = np.concatenate([X_rna, path_for_models], axis=1).astype(np.float32)

    # 5th fusion: concat->SVD(z_dim)
        try:
            svd_concat = TruncatedSVD(n_components=args.z_dim, random_state=args.seed)
            Z_concat = svd_concat.fit_transform(X_concat).astype(np.float32)
            save_with_embedding(
                adata_run, Z_concat, "X_fused_concat",
                os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.concat.h5ad")
            )
        except Exception as e:
            print(f"[WARN] Failed to save concat embedding for {tag}: {e}")

    # Early fusion AE
        Z_early = train_early_ae(
            X_concat, args.hidden, args.z_dim, args.epochs, args.batch_size,
            args.lr, args.weight_decay, device
        )
        save_with_embedding(
            adata_run, Z_early, "X_fused_early",
            os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.early_ae.h5ad")
        )

    # Mid fusion AE
        Z_mid = train_mid_ae(
            X_rna, path_for_models, args.hidden, args.z_dim, args.epochs, args.batch_size,
            args.lr, args.weight_decay, device
        )
        save_with_embedding(
            adata_run, Z_mid, "X_fused_mid",
            os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.mid_ae.h5ad")
        )

    # Late fusion AE
        Z_late = train_late_ae(
            X_rna, path_for_models, args.hidden, args.z_dim, args.epochs, args.batch_size,
            args.lr, args.weight_decay, device
        )
        save_with_embedding(
            adata_run, Z_late, "X_fused_late",
            os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.late_ae.h5ad")
        )

    # CLIP-like contrastive
        Z_clip = train_clip(
            X_rna, path_for_models, args.hidden, args.z_dim, args.epochs, args.batch_size,
            args.lr, args.weight_decay, args.temperature, device
        )
        save_with_embedding(
            adata_run, Z_clip, "X_fused_clip",
            os.path.join(args.out_dir, f"{args.out_prefix}.{tag}.clip.h5ad")
        )

        print(f"[DONE] {tag} saved combined + 5 embeddings")

        print("DONE")
    if last_combined_path is not None:
        print(f"Last saved combined: {last_combined_path}")

if __name__ == "__main__":
    main()