import scanpy as sc
import pandas as pd

from .sample_constants import PANEL_2_COLLECTIONIDX_TO_SAMPLE, PANEL_1_COLLECTIONIDX_TO_SAMPLE

def to_points_df(
    adata: sc.AnnData,
    label_col: str,
    panel: int | None = None,
    sample_col: str = 'Sample',
    growth_col: str | None = 'Growth pattern',
) -> pd.DataFrame:
    """Convert an AnnData object to a standardized points DataFrame.

    The returned table contains spatial coordinates (`x`, `y`), a label column
    (`Label`), collection index, sample, and growth pattern. If sample or growth
    information is missing in `adata.obs`, values are inferred from
    `collectionIndex` and panel-specific mappings.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData with spatial coordinates in `adata.obsm["spatial"]`.
    label_col : str
        Column in `adata.obs` to use as point labels.
    panel : int | None, optional
        Panel identifier (`1` or `2`) used to map `collectionIndex` to sample
        names when `sample_col` is not available.
    sample_col : str, optional
        Preferred sample column in `adata.obs`.
    growth_col : str | None, optional
        Preferred growth-pattern column in `adata.obs`.

    Returns
    -------
    pd.DataFrame
        A filtered and typed points DataFrame suitable for downstream analysis.

    Raises
    ------
    ValueError
        If `panel` is invalid, or required metadata cannot be resolved.
    """

    if panel is not None:
        if panel == 2:
            collection_to_sample = PANEL_2_COLLECTIONIDX_TO_SAMPLE
        elif panel == 1:
            collection_to_sample = PANEL_1_COLLECTIONIDX_TO_SAMPLE
        else:
            raise ValueError(f"Invalid panel number: {panel}. Expected 1 or 2.")

    if sample_col in adata.obs.columns:
        sample_values = adata.obs[sample_col].values
    elif "collectionIndex" in adata.obs.columns:
        sample_values = adata.obs["collectionIndex"].map(collection_to_sample).values
    else:
        raise ValueError(f"Neither '{sample_col}' nor 'collectionIndex' found in adata.obs.columns")

    if growth_col in adata.obs.columns:
        growth_values = adata.obs[growth_col].values
    elif "collectionIndex" in adata.obs.columns:
        growth_values = ["EHGP" if "ENC" in sample else "RHGP" for sample in sample_values]
    else:
        raise ValueError(f"Neither '{growth_col}' nor 'collectionIndex' found in adata.obs.columns")


    
    df = pd.DataFrame(
        {
            "x": adata.obsm["spatial"][:, 0],
            "y": adata.obsm["spatial"][:, 1],
            "Label": adata.obs[label_col],
            "collectionIndex": adata.obs["collectionIndex"].values,
            "Sample": sample_values,
            "Growth pattern": growth_values,
        }
    ).astype(
        {
            "x": "uint16",
            "y": "uint16",
            "Label": "category",
            "collectionIndex": "category",
            "Sample": "category",
            "Growth pattern": "category",
        }
    )

    return df

if __name__ == "__main__":
    import scanpy as sc
    from pathlib import Path
    image_folder = Path("data/images")

    # Load anndata objects with reads and cells.
    panel = 1
    reads_ad = sc.read_h5ad(f"data/files/prepared_reads_panel_{panel}.h5ad")
    cells_ad = sc.read_h5ad(f"prepared_adata/prepared_cells_panel_{panel}.h5ad")

    # Convert to standardized dataframes with columns "x", "y", "Label", "Sample", "Growth pattern" 
    # for downstream zonation analysis.
    reads = to_points_df(reads_ad, panel=panel, label_col="Gene")