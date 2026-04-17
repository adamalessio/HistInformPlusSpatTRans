# HIPSTR

**Histology-Informed Spatial Transcriptomic Representations (HIPSTR)**

This repository contains the analysis code used for the manuscript:

> **Multimodal representations mitigate failure modes in spatial transcriptomic clustering under degradation**

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

The manuscript Data Availability section should point readers to the exact public URLs and DOI.

## What to include in the GitHub release

For the public repository, include:

- the code in this repository
- a release tag corresponding to the manuscript version
- the final `README.md`
- a populated `Code availability` statement in the manuscript with the GitHub URL
- example commands or a small test dataset if redistribution is allowed

You do **not** need to upload large intermediate `.h5ad` files or full raw datasets to GitHub. Instead:
- provide acquisition links in the README
- provide commands to regenerate results
- archive processed outputs separately on Zenodo if needed

## Recommended manuscript reproducibility statement

> All code for preprocessing, degradation simulation, radiomics extraction, multimodal fusion, clustering, evaluation, and figure generation is available at [GitHub URL]. A frozen release corresponding to the manuscript version is archived at [Zenodo DOI if created].

## Notes for Genome Biology submission

To maximize reproducibility for peer review:

- keep the package installable with `pip install -e .`
- tag the commit used for submission
- consider creating a GitHub release and linking it to Zenodo for a versioned DOI
- replace placeholder URLs in `pyproject.toml` and this README with the final public repository URL
- keep any environment-specific paths out of committed scripts where possible

## Current packaging approach

This repository is already **pip-installable**. The current package exposes the manuscript scripts as console entry points using lightweight wrappers. This keeps your original analysis scripts intact while making the codebase easier to install and run.

A later refactor could move core logic into importable modules, but that is not required for manuscript submission.

## Citation

If you use this code, please cite the associated manuscript once available.
