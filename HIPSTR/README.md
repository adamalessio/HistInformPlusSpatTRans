# HIPSTR

**Histology-Informed Spatial Transcriptomic Representations (HIPSTR)**

This repository contains the analysis code used for the manuscript:

> **Evaluating Multimodal Clustering Robustness to Spatial Transcriptomics Data Degradation**

The repository is organized for reproducibility and journal submission. It includes code for:

- radiomics feature extraction from Visium-aligned histology images
- multimodal fusion of RNA and radiomics features
- simulation of transcriptomic degradation (ambient contamination, dropout, and related perturbations)
- clustering evaluation using agreement, geometry, failure-mode, and marker-preservation metrics
- figure generation for the manuscript
- baseline comparison workflows including SpaGCN support

## Repository structure

```text
HIPSTR/
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── LICENSE
├── src/
│   └── hipstr/
│       ├── __init__.py
│       ├── _runner.py
│       ├── cli/
│       └── scripts/
├── scripts/
│   ├── run_pipeline.sh
│   └── original/
├── config/
│   └── example_config.yaml
├── docs/
└── examples/
```

## Installation

### Option 1: pip install from a local clone

```bash
git clone https://github.com/USERNAME/HIPSTR.git
cd HIPSTR
pip install -e .
```

### Option 2: conda environment

```bash
conda env create -f environment.yml
conda activate hipstr
pip install -e .
```

## Command-line entry points

Installing the package exposes the following commands:

- `hipstr-fuse`
- `hipstr-evaluate`
- `hipstr-plot-breast`
- `hipstr-plot-overlays`
- `hipstr-run-spagcn`
- `hipstr-concat-spagcn`
- `hipstr-radiomics-legacy`

These commands wrap the manuscript scripts packaged under `src/hipstr/scripts/`.

## Minimal workflow

A typical workflow is:

1. Generate radiomics and fused embeddings
2. Run degradation and clustering pipelines
3. Evaluate outputs against baseline and pathologist annotations
4. Reproduce figures

Example wrapper script:

```bash
bash scripts/run_pipeline.sh
```

## Data required

This study uses publicly available spatial transcriptomics datasets:

- **Human dorsolateral prefrontal cortex (DLPFC)** from the `spatialLIBD` resource
- **Human Breast Cancer Block A** from Zenodo / 10x-compatible Visium resources



## Citation

If you use this code, please cite the associated manuscript once available.
