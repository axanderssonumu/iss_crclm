import ast
import json
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Literal
from rasterio.features import rasterize
from rasterio.transform import Affine

from .sample_constants import PANEL_2_COLLECTIONIDX_TO_SAMPLE, PANEL_2_IM_WIDTH_HEIGHT, PANEL_1_COLLECTIONIDX_TO_SAMPLE, PANEL_1_IM_WIDTH_HEIGHT



def _parse_classification(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return None


def _rgb_to_hex(color):
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r = max(0, min(255, int(color[0])))
            g = max(0, min(255, int(color[1])))
            b = max(0, min(255, int(color[2])))
            return f"#{r:02X}{g:02X}{b:02X}"
        except (TypeError, ValueError):
            return pd.NA
    return pd.NA


def load_geojson(path, panel=2):
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])

    if "classification" in gdf.columns:
        parsed_classification = gdf["classification"].map(_parse_classification)
        gdf["classification"] = parsed_classification.map(
            lambda entry: entry.get("name", pd.NA) if isinstance(entry, dict) else pd.NA
        )
        gdf["color"] = parsed_classification.map(
            lambda entry: _rgb_to_hex(entry.get("color")) if isinstance(entry, dict) else pd.NA
        )

    if {"name", "classification"}.issubset(gdf.columns):
        mask = gdf["name"].eq("My region") & gdf["classification"].notna()
        gdf.loc[mask, "name"] = gdf.loc[mask, "classification"]

    rename_map = {
        "classification": "Region",
        "name": "Subregion",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    preferred = ["geometry", "Subregion", "Region", "color", "collectionIndex"]
    present = [col for col in preferred if col in gdf.columns]
    if present:
        gdf = gdf[present + [col for col in gdf.columns if col not in present]]

    if panel == 2:
        gdf["Sample"] = gdf["collectionIndex"].map(PANEL_2_COLLECTIONIDX_TO_SAMPLE).astype("category")
        gdf["Image Size"] = [PANEL_2_IM_WIDTH_HEIGHT[s] for s in gdf["Sample"]]
    elif panel == 1:
        gdf["Sample"] = gdf["collectionIndex"].map(PANEL_1_COLLECTIONIDX_TO_SAMPLE).astype("category")
        gdf["Image Size"] = [PANEL_1_IM_WIDTH_HEIGHT[s] for s in gdf["Sample"]]
    else:
        raise ValueError(f"Invalid panel number: {panel}. Expected 1 or 2.")
    return gdf




def load_distance_label_mask(gdf: gpd.GeoDataFrame, sample: str, region: Literal['Liver inside', 'Liver outside', 'Tumor inside', 'Tumor outside'], max_distance: float = 300):
    if max_distance <= 0:
        raise ValueError("max_distance must be > 0")

    step = 50
    distance_limit = int(max_distance)

    gdf_region = gdf[(gdf['Sample'] == sample) & (gdf['Region'] == region)]
    if gdf_region.empty:
        raise ValueError(f"No rows found for sample='{sample}' and region='{region}'")

    width, height = gdf_region['Image Size'].iloc[0]
    width = int(width)
    height = int(height)
    label_mask = np.zeros((height, width), dtype=np.uint16)

    subregion_keys = [
        f'{region} {lower} μm - {lower + step} μm'
        for lower in range(0, distance_limit, step)
    ]

    shapes = []
    for idx, subregion in enumerate(subregion_keys, start=1):
        geoms = gdf_region.loc[gdf_region['Subregion'] == subregion, 'geometry'].dropna()
        if geoms.empty:
            continue

        for geom in geoms:
            if geom.is_empty:
                continue
            shapes.append((geom, idx))

    if not shapes:
        return label_mask

    # Identity transform maps pixel indices directly to geometry coordinates.
    label_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=0,
        out=label_mask,
        transform=Affine.identity(),
        all_touched=False,
        dtype=np.uint16,
    )

    return label_mask



def get_counts_per_area(df, xy_cols, label_col, sample_col, gdf: gpd.GeoDataFrame, max_distance: float = 300):

    df = df.copy()
    region_labels = ['Liver inside', 'Liver outside', 'Tumor inside', 'Tumor outside']

    # Precreate distance columns and area lookup
    for col in region_labels:
        df[col] = 0
    region_areas = {}

    for sample, sample_reads in df.groupby(sample_col, observed=True):
        print(f"Processing sample {sample}")
        y = sample_reads[xy_cols[1]].to_numpy()
        x = sample_reads[xy_cols[0]].to_numpy()
        for region_type in region_labels:
            distance_label_mask = load_distance_label_mask(gdf, sample, region_type, max_distance=max_distance)
            df.loc[sample_reads.index, region_type] = distance_label_mask[y, x]
            area_px = (distance_label_mask > 0).sum()
            area_um2 = area_px * (0.16**2)  # convert from pixels to mm^2, as each pixel is 0.16mm x 0.16mm
            area_mm2 = area_um2 / 1e6
            region_areas[(sample, region_type)] = float(area_mm2)

    # Remove reads that are in none of the regions
    df = df.loc[~df[region_labels].eq(0).all(axis=1)].copy()

    df['Regions'] = (
        df[region_labels]
        .replace(0, np.inf)
        .idxmin(axis=1)
        .astype('category')
    )

    # Keep only needed columns
    df = df.drop(columns=region_labels)

    # Count reads per sample/gene/region
    # Index = Sample, Gene; Columns = Regions; Values = counts per area
    region_counts = (
        df.groupby([sample_col, label_col, 'Regions'], observed=True)
        .size()
        .rename('Counts')
        .reset_index()
    )

    # Build matching region-area table and compute counts per area
    area_df = pd.Series(region_areas, name='Area').rename_axis([sample_col, 'Regions']).reset_index()
    area_df = area_df.groupby([sample_col, 'Regions'], as_index=False, observed=True)['Area'].sum()

    region_counts = region_counts.merge(area_df, on=[sample_col, 'Regions'], how='left')
    region_counts['Counts per area'] = region_counts['Counts'] / region_counts['Area'].replace(0, np.nan)

    # We now have indices Sample and Gene, and columns = Regions, with values = counts per area. 
    region_counts = region_counts.pivot(index=[sample_col, label_col], columns='Regions', values='Counts per area').fillna(0)

    return region_counts
    # Normalize counts per area for each gene and sample to the total counts per area across regions
    # This is needed to compare the relative distribution of reads across regions for each gene and sample, 
    # as the total counts per area can differ between samples and genes.
