# visium_radiomics_banksy_clustering.py

import os
import numpy as np
import pandas as pd
import scanpy as sc
from radiomics import featureextractor
import squidpy as sq
import tifffile
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


h5_path = '/home/ilovele1/breast_cancer/section1/filtered_feature_bc_matrix.h5'
parquet_path = '/home/ilovele1/breast_cancer/section1/spatial/tissue_positions_list.csv'
he_path = '/home/ilovele1/breast_cancer/section1/Visium_Human_Breast_Cancer_image.tif'
sample_id = 'Brain'


# ------------------------------
# Load Visium data and H&E
# ------------------------------
def load_visium_sample(h5_path, parquet_path, he_path, sample_id, top_genes=50):
    adata = sc.read_10x_h5(h5_path)
    #adata = sq.read.visium(path=os.path.dirname(h5_path), count_file=os.path.basename(h5_path))
    #pos = pd.read_parquet(parquet_path, engine="auto")
    pos = pd.read_csv(parquet_path, delimiter = ',', header = None)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    adata.obs = adata.obs.merge(pos.set_index("barcode"), left_index=True, right_index=True)
    adata.obs["x"] = adata.obs["pxl_col"]
    adata.obs["y"] = adata.obs["pxl_row"]
    adata.obs["sample_id"] = sample_id
    sc.pp.highly_variable_genes(adata, n_top_genes=top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    he_img = tifffile.imread(he_path)
    #he_img = np.transpose(he_img, (1, 2, 0))   # now (24240, 24240, 3)
    #he_img = (he_img / 256).astype(np.uint8)
    he_img = Image.fromarray(he_img)
    return adata, he_img

# ------------------------------
# Extract radiomics features
# ------------------------------
def extract_radiomics_for_spots(adata, he_image, patch_size=400):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    features = []
    for _, row in adata.obs.iterrows():
        x, y = int(row["pxl_col"]), int(row["pxl_row"])
        patch = he_image.crop((x - patch_size//2, y - patch_size//2, x + patch_size//2, y + patch_size//2))
        patch = patch.convert("L")
        np_patch = np.array(patch)
        mask = np.ones_like(np_patch)
        #yy, xx = np.ogrid[:patch_size, :patch_size]
        #center = patch_size // 2
        #disk_mask = (xx - center)**2 + (yy - center)**2 <= (patch_size / 2)**2
        #mask = disk_mask.astype(np.uint8)
        img_sitk = sitk.GetImageFromArray(np_patch)
        mask_sitk = sitk.GetImageFromArray(mask)
        feats = extractor.execute(img_sitk, mask_sitk)
        feats = {k: v for k, v in feats.items() if "diagnostics" not in k}
        features.append(feats)
    return pd.DataFrame(features, index=adata.obs_names)


# ------------------------------
# Main pipeline
# ------------------------------
def run_pipeline(h5_path, parquet_path, he_path, sample_id):
    adata, he_img = load_visium_sample(h5_path, parquet_path, he_path, sample_id)
    radiomics_df = extract_radiomics_for_spots(adata, he_img)
    radiomics_df = radiomics_df.loc[adata.obs_names]
    full_matrix = np.hstack([ StandardScaler().fit_transform(radiomics_df.values)])
    embedding = PCA(n_components=30).fit_transform(full_matrix)
    return adata, full_matrix, embedding, radiomics_df




