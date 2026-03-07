# Python Notebooks from  Spatial transcriptomics profiling of histopathological growth patterns in colorectal cancer liver metastases

This repository contains code used to reproduce the figures and analyses presented in the paper:

> *Spatial transcriptomics profiling of histopathological growth patterns in colorectal cancer liver metastases*

Analyses are implemented in a single Jupyter notebook:

**➡ [`analyses.ipynb`](analyses.ipynb)**

The notebook contains the methods used for clustering the data and generating the analysis figures presented in the manuscript.

## Online Viewer and Data availability

The processed data required to run the notebook is not included in this repository due to file size limitations.

The data can be interactively be visualized from:

**➡ [`Online viewer`](https://crclm2.serve.scilifelab.se)**

The data, including images, gene count tables (AnnData), raw transcripts (AnnData) and regions (GeoJSON) can be downloaded from:

**➡ [`Panel 2 data (.tar)`](https://crclm2.serve.scilifelab.se/iss_panel_2.tmap?dl=1)**

**➡ [`Panel 1 data (.tar)`](https://crclm2.serve.scilifelab.se/iss_panel_1.tmap?dl=1)**

After downloading, extract .tar data and place it in a `data/` directory in the repository so that the folder structure looks like:

```
project_root/
│
├── notebooks/
│   └── reproduce_figures.ipynb
│
├── data/
│   ├── files/
│   │   └── prepared_cells_panel_2.h5ad
│   │   └── prepared_reads_panel_2.h5ad
│   │   └── regions_panel_2.json
│   │
│   ├── images/
│   │   └── iss_panel_<SAMPLE ID>.tif
│   │   └── ...
└── README.md
```


## Environment setup

The analysis was performed using Python 3. To recreate the environment used in this repository:

1. Create a new Conda environment:

```bash
conda create -n iss_crclm python=3.10
conda activate iss_crclm
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

This will hopefully install all packages used in the notebook and analysis.
