import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import offset_copy
from typing import Optional, List

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator

plt.rcParams['svg.fonttype'] = 'none'

# GLOBALS
BOXPLOT_FONTSIZE = 8
BOXPLOT_HEIGHT = 2.25
BOXPLOT_WIDTH = lambda N: max(1.2, N * 0.4)


def custom_dotplot_raw(
        adata: sc.AnnData, 
        cluster_key: str, 
        genes: Optional[List[str]] =None,
        cluster_order: Optional[List[str]] =None, 
        plot_marker_genes: bool =False, 
        figsize: Optional[tuple] =None, 
        dotmax: float =0.8, 
        save: Optional[str] =None
    ):
    """
    Create a dotplot visualization of raw gene expression across clusters.
    
    Generates a dotplot where each dot represents the mean expression of a gene 
    in a cluster. Dot size indicates the percentage of cells expressing the gene, 
    and color intensity represents the scaled mean expression value. Per-gene 
    colorbars show the expression range from zero to the largest mean expression among clusters.
    
    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object with raw counts in adata.raw or adata.layers['X_raw'].
    cluster_key : str
        Key in adata.obs containing cluster labels.
    genes : Optional[List[str]], default=None
        List of gene names to plot. If None, uses all marker genes from 
        adata.uns['marker_genes'] for all clusters in cluster_order.
    cluster_order : Optional[List[str]], default=None
        Order of clusters to display on y-axis. If None, uses categorical order 
        from adata.obs[cluster_key].
    plot_marker_genes : bool, default=False
        If True, marks marker genes (from adata.uns['marker_genes']) with 'x' symbols.
    figsize : Optional[tuple], default=None
        Figure size (width, height) in inches. If None, automatically sizes based 
        on number of genes and clusters.
    dotmax : float, default=0.8
        Maximum dot size used in the dotplot.
    save : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.
    
    Returns
    -------
    None
        Modifies adata.layers in place and displays/saves the figure.
    
    Examples
    --------
    >>> ad = sc.read_h5ad('data.h5ad')
    >>> custom_dotplot_raw(ad, cluster_key='Clusters', plot_marker_genes=True)
    
    >>> custom_dotplot_raw(
    ...     ad, 
    ...     cluster_key='Clusters', 
    ...     genes=['CD8A', 'CD4', 'FOXP3'],
    ...     cluster_order=['T cells', 'B cells', 'Myeloid'],
    ...     save='dotplot.png'
    ... )
    """
   

    if cluster_order is None:
        cluster_order = adata.obs[cluster_key].cat.categories.tolist()
    if genes is None:
        if 'marker_genes' not in adata.uns:
            raise ValueError("If genes is None, adata.uns must contain 'marker_genes'")
        else:
            if 'marker_genes' not in adata.uns:
                raise ValueError("If genes is None, adata.uns must contain 'marker_genes'")
            genes = []
            for cluster in cluster_order:
                genes.extend([g for g in adata.uns['marker_genes'][cluster]])
            genes = list(genes)
 
    if figsize is None:
        width = max(len(genes) * 0.4 + 1, 4)
        height = len(cluster_order) * .25 + 1
    else:
        width, height = figsize


    if 'X_raw' not in adata.layers:
        if adata.raw is None:
            raise ValueError("adata must have raw counts in adata.raw or adata.layers['X_raw']")
        else:
            adata.layers['X_raw'] = adata.raw.X.toarray() if sp.issparse(adata.raw.X) else adata.raw.X.copy()
    else:
        if sp.issparse(adata.layers['X_raw']):
            adata.layers['X_raw'] = adata.layers['X_raw'].toarray()


    # Subset the data to the specified clusters and genes, and compute average expression per cluster
    cells = adata[adata.obs[cluster_key].isin(cluster_order)].copy()
    cells.obs[cluster_key] = cells.obs[cluster_key].cat.remove_unused_categories() 

    # Compute average expression per cluster for the specified genes, 
    # and scale by the maximum average expression across clusters for each gene
    n_clusters = len(cells.obs[cluster_key].cat.categories)
    avg_expression = np.zeros((n_clusters, cells.layers['X_raw'].shape[1]))
    for ci, clust in enumerate(cells.obs[cluster_key].cat.categories):
        mask = cells.obs[cluster_key] == clust
        counts = cells.layers['X_raw'][mask,:]
        avg_expression[ci,:] = np.mean(counts, axis=0)
    max_expression = avg_expression.max(axis=0)
    
    X_scaled = cells.layers['X_raw'].copy()
    X_scaled = X_scaled / max_expression  # scale each gene by its max average expression across clusters
    cells.layers['X_raw_scaled'] = X_scaled
 
    # Make dotplot
    bwr = plt.get_cmap('bwr')
    custom_cmap = LinearSegmentedColormap.from_list('custom_bwr', bwr(np.linspace(0.5, 1, 5)))
    fig = sc.pl.dotplot(
        cells, 
        genes, 
        cluster_key,     
        vmin=0,
        vmax=1,
        figsize=(width, height),
        dot_max=dotmax,
        categories_order=cluster_order,
        cmap=custom_cmap,
        size_title='% of cells in cluster\nexpressing the gene',
        layer='X_raw_scaled',
        return_fig=True
    )
 
    ax_dotplot = fig.get_axes()['mainplot_ax']
    cbar = fig.get_axes()['color_legend_ax']
    cbar.remove()
 
    # Gene labels in italic
    for label in ax_dotplot.get_xticklabels():
        label.set_fontstyle('italic')
 
    for label_idx, label in enumerate(ax_dotplot.get_yticklabels()):
        label.set_text(cluster_order[label_idx])
 
    n_genes = len(genes)
 
    dot_y, dot_x = np.indices((n_clusters, n_genes))
    dot_y = dot_y + 0.5
    dot_x = dot_x + 0.5
 
    # Draw an X on marker genes
    if plot_marker_genes:
        for gi, gene in enumerate(genes):
            for ci, cluster in enumerate(cluster_order):
                if gene in cells.uns['marker_genes'][cluster]:
                    ax_dotplot.scatter(dot_x[ci, gi], dot_y[ci, gi], marker='x', color='black', s=14)
 
 

    # Make a bar for each gene showing the color scale, with a fixed height in pixels, and labels for 0 and max average expression.
    # before the gene loop, pick a fixed height in pixels:
    desired_bar_height_px = 25  # adjust to taste
    fig_height_inch = fig.fig.get_size_inches()[1]
    dpi = fig.fig.dpi
    # convert to figure fraction:
    bar_height_frac = desired_bar_height_px / (fig_height_inch * dpi)  # constant physical height
 
 
    # desired vertical padding in pixels
    desired_padding_px = 10
 
    fig_height_inch = fig.fig.get_size_inches()[1]
    dpi = fig.fig.dpi
    # convert to figure fraction
    pad_frac = desired_padding_px / (fig_height_inch * dpi)
 
    # then when placing each colorbar:
    desired_bar_width_px = 10
    fig_width_inch = fig.fig.get_size_inches()[0]
    bar_width_frac = desired_bar_width_px / (fig_width_inch * dpi)
 
    for idx, gene in enumerate(genes):
        # Extract data for the specific gene
        k = np.where(gene == cells.var_names)[0]
        max_value = max_expression[k][0]
 
        # inside your loop over genes, replace bar_height and creation with: 
        # position calculations as before:
        x_data = idx + 0.5
        y_data = ax_dotplot.get_ylim()[1]  # top of the axis
 
        x_disp, _ = ax_dotplot.transData.transform((x_data, y_data))
        x_fig, _ = fig.fig.transFigure.inverted().transform((x_disp, 0))
 
        bar_left = x_fig - bar_width_frac / 2
        bar_bottom = ax_dotplot.get_position().y1 + pad_frac  # still a small gap above main plot
 
        # create colorbar axis with fixed height in figure coords
        colorbar_ax = fig.fig.add_axes([bar_left, bar_bottom, bar_width_frac, bar_height_frac])
 
        # Create a colorbar
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array(np.linspace(0, 1.0))
 
        cbar = plt.colorbar(sm, cax=colorbar_ax, orientation='vertical')
        # Hide the default ticks and labels
        cbar.set_ticks([])
        cbar.ax.tick_params(length=0)  # remove tick lines too
 
        # Get bar position in axis coordinates
        x_center = 0.5  # centered along x
        y_top = 1.05    # just above top
        y_bottom = -0.05  # just below bottom
 
        # Add custom labels
        cbar.ax.text(x_center, y_top, f'{max_value:0.1f}', ha='center', va='bottom', fontsize=7, transform=cbar.ax.transAxes)
        cbar.ax.text(x_center, y_bottom, '0.0', ha='center', va='top', fontsize=7, transform=cbar.ax.transAxes)
 
 
    if plot_marker_genes:
        # Position marker just outside the right edge of the plot using pixel offset
        x, y = 1.0, 0.05  # right edge of axes
    
        # Add label with same offset
        offset_trans = offset_copy(ax_dotplot.transAxes, fig=fig.fig, x=16, y=0, units='points')
        ax_dotplot.text(x, y + 0.05, 'Marker genes', ha='left', va='bottom',
                        transform=offset_trans, fontsize=9)
 
        # Add marker with offset transform
        offset_trans = offset_copy(ax_dotplot.transAxes, fig=fig.fig, x=44, y=0, units='points')
        ax_dotplot.plot(x, y, marker='x', color='black', transform=offset_trans,
                        markersize=10, clip_on=False)
    # Define a pixel-based offset transform (relative to the axes)
    offset = offset_copy(ax_dotplot.transAxes, fig=ax_dotplot.figure,
                        x=-7, y=20, units='points')
 
    ax_dotplot.text(0, 1, 'Mean count\nin cluster',
                    transform=offset,
                    ha='right', va='center',
                    weight='normal', size=9)
 
    if save is not None:
        fig.fig.savefig(save, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.3)






def force_same_axis_height(ax, height=0.8, bottom=0.1):
    """
    Force an axis to have the same height in figure coordinates.
    Width is automatically adjusted to accommodate ticklabels.
    """
    fig = ax.figure
    # Get the current axis position
    pos = ax.get_position()

    # Keep the same left coordinate (so ticklabels don’t get clipped),
    # but force identical height.
    new_pos = [pos.x0, bottom, pos.width, height]
    ax.set_position(new_pos)



def compute_p_values(df, clusters, regions, groups):
    """Compute adjusted Mann-Whitney U test p-values between growth patterns"""
    records = []

    for region in regions:
        for cluster in clusters:
            g1 = df.query('Cluster == @cluster and Region == @region and Group == @groups[0]')['Proportion (%)']
            g2 = df.query('Cluster == @cluster and Region == @region and Group == @groups[1]')['Proportion (%)']
            p = mannwhitneyu(g1, g2, alternative='two-sided').pvalue if len(g1) and len(g2) else 1.0
            records.append({'Region': region, 'Cluster': cluster, 'P-value': p})

    df_p = pd.DataFrame(records)
    df_p['P-value adjusted'] = multipletests(df_p['P-value'], method='fdr_bh')[1]
    return {(r['Cluster'], r['Region']): r['P-value adjusted'] for _, r in df_p.iterrows()}


def plot_cluster_proportions_between_groups(adata: sc.AnnData, cluster_key: str, group_key: str, sample_key: str, patient_key: str, region_mask_keys: List[str], clusters_to_plot: Optional[List[str]] = None, save: Optional[str] =None, plot_horizontal: bool =True, pval_format:str='star'):
    """
    Plot cluster proportions between two groups across multiple regions.
    
    Creates boxplots showing the proportion of cells in each cluster within 
    different tissue regions, separated by two groups. Performs Mann-Whitney U 
    tests with FDR correction and annotates significant differences on the plots.
    
    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object with cluster and group annotations.
    cluster_key : str
        Key in adata.obs containing cluster labels (must be categorical).
    group_key : str
        Key in adata.obs containing exactly 2 group labels (must be categorical).
        Example: 'Growth_pattern' with values ['EHGP', 'RHGP'].
    sample_key : str
        Key in adata.obs containing sample identifiers (must be categorical).
    patient_key : str
        Key in adata.obs containing patient identifiers (must be categorical).
    region_mask_keys : List[str]
        List of boolean column names in adata.obs that define tissue regions.
        Each column should contain True/False indicating membership in that region.
        Example: ['Liver', 'Tumor', 'Stroma'].
    clusters_to_plot : Optional[List[str]], default=None
        List of cluster to plot
    save : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.
    plot_horizontal : bool, default=True
        Whether to plot the clusters horizontally (True) or vertically (False).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of subplot axes, one per cluster.
    
    Raises
    ------
    ValueError
        If cluster_key, group_key, or sample_key not in adata.obs.
        If any of these keys are not categorical dtype.
        If group_key does not have exactly 2 unique values.
    
    Notes
    -----
    - Proportions are calculated per sample and region, then averaged across 
      samples to avoid pseudo-replication. Proportions from samples of the same patient are
        averaged together to avoid inflating n.
    - Statistical tests: Mann-Whitney U test with FDR-corrected p-values (fdr_bh).
    - Significance thresholds: * p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001.
    - Color scheme: '#6497bf' for first group, '#d8031c' for second group.
    
    Examples
    --------
    >>> ad = sc.read_h5ad('data.h5ad')
    >>> fig, axs = compare_cluster_proportions_between_groups(
    ...     ad,
    ...     cluster_key='CellType',
    ...     group_key='Treatment',
    ...     sample_key='Sample',
    ...     patient_key='Patient',
    ...     region_mask_keys=['Tumor', 'Stroma', 'Margin'],
    ...     save='cluster_comparison.png'
    ... )
    """
    unique_regions = region_mask_keys
    
    # Check that columns are categorical
    for key in [cluster_key, group_key, sample_key, patient_key]:
        if key not in adata.obs:
            raise ValueError(f"{key} not found in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[key]):
            raise ValueError(f"{key} must be a categorical column in adata.obs")
    
    unique_clusters = adata.obs[cluster_key].cat.categories
    unique_samples = adata.obs[sample_key].cat.categories
    unique_groups = adata.obs[group_key].cat.categories
    unique_patients = adata.obs[patient_key].cat.categories

    if len(unique_groups) != 2:
        raise ValueError(f"Expected exactly 2 unique groups in {group_key}, but found {len(unique_groups)}: {unique_groups}")
    palette = {unique_groups[0]: '#6497bf', unique_groups[1]: '#d8031c'}
    

    # For each sample, region, and cluster, calculate the proportion of cells in that sample and region that belong to that cluster. 
    # Store results in a DataFrame with columns: Sample, Region, Cluster, Group, Proportion (%)
    records = []
    for sample in unique_samples:
        obs_by_sample = adata.obs[adata.obs[sample_key] == sample]
        group = obs_by_sample[group_key].iloc[0] if len(obs_by_sample) else 'Unknown'
        patient = obs_by_sample[patient_key].iloc[0] if len(obs_by_sample) else 'Unknown'
        for region in unique_regions:
            obs_by_region = obs_by_sample[obs_by_sample[region] == True] 
            if not len(obs_by_region):
                continue
            
            for cluster in unique_clusters:
                obs_by_cluster = obs_by_region[obs_by_region[cluster_key] == cluster]
                freq = (len(obs_by_cluster) / len(obs_by_region)) * 100 if len(obs_by_region) else 0
                records.append({
                    'Sample': sample,
                    'Patient': patient,
                    'Region': region,
                    'Cluster': cluster,
                    'Group': group,
                    'Proportion (%)': freq
                })

    df = pd.DataFrame(records).groupby(['Patient', 'Region', 'Cluster', 'Group'])['Proportion (%)'].mean().reset_index()
    adjusted_pval_dict = compute_p_values(df, unique_clusters, unique_regions, unique_groups)

    # Create the figure and axes
    if clusters_to_plot is None:
        clusters_to_plot = unique_clusters

    # Plotting
    n_subplots = len(clusters_to_plot)
    if plot_horizontal:
        height = BOXPLOT_HEIGHT
        width = BOXPLOT_WIDTH(len(unique_regions)) * n_subplots
        fz = BOXPLOT_FONTSIZE
        fig, axs = plt.subplots(ncols=n_subplots, nrows=1, figsize=(width, height), sharex=False)
    else:
        height = BOXPLOT_HEIGHT * n_subplots
        width = BOXPLOT_WIDTH(len(unique_regions))
        fz = BOXPLOT_FONTSIZE
        fig, axs = plt.subplots(ncols=1, nrows=n_subplots, figsize=(width, height), sharex=True, constrained_layout=True)

    if n_subplots == 1:
        axs = np.array([axs])  # ensure axs is always an array for consistent indexing
        
    for ax, cluster_to_plot in zip(axs, clusters_to_plot):

        data = df.query('Cluster == @cluster_to_plot')
        g = sns.boxplot(
            data=data, 
            y='Proportion (%)', 
            x='Region', 
            hue='Group', 
            order=unique_regions, 
            palette=palette, 
            ax=ax
        )

        sns.despine()

        # Style adjustments
        ax.set_xlabel(None)
        if plot_horizontal and ax == axs[0]:
            ax.set_ylabel('Proportion (%)', fontsize=fz)
        else:
            ax.set_ylabel(None)

        ax.tick_params(axis='x', labelsize=fz, rotation=90)
        ax.tick_params(axis='y', labelsize=fz)
        ax.set_title(cluster_to_plot, fontsize=fz)
        
        # Only three y ticks, rounded to integers
        yticks = np.round(np.linspace(0, np.max(ax.get_ylim()), 3)).astype(int)
        ax.set_yticks(yticks)

        # Add vertical grid lines between regions
        xticks = ax.get_xticks()
        for x in [(xticks[i] + xticks[i + 1]) / 2 for i in range(len(xticks) - 1)]:
            ax.axvline(x, color='gray', linestyle='--', alpha=0.3, zorder=0)

        # Add significance annotations
        pairs = []
        p_values = []
        for region in unique_regions:
            region_data = data.query(f'Region == @region')
            # only annotate if the mean proportion in at least one group is >3% to avoid cluttering with insignificant comparisons
            if region_data.groupby('Group')['Proportion (%)'].median().max() > 3:
                pairs.append(((region, unique_groups[0]), (region, unique_groups[1])))
                p_values.append(adjusted_pval_dict.get((cluster_to_plot, region), 1.0))
                
        if pairs:
            annotator = Annotator(ax, pairs, data=data, hue='Group', y='Proportion (%)', x='Region', order=unique_regions)
            annotator.configure(
                test=None,
                text_format=pval_format,
                hide_non_significant=True if pval_format == 'star' else False,
                loc='inside',
                pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"]]
            ).set_pvalues_and_annotate(p_values)
        
        g.legend(frameon=False, loc='best',  ncols=1, fontsize=fz)
        
        if plot_horizontal or n_subplots == 1:
            force_same_axis_height(ax)

    if not plot_horizontal:
        for ax in axs:
            ax.set_box_aspect(1) 


    if save is not None:
        fig.savefig(save, dpi=300, transparent=True, bbox_inches='tight')

    return fig, axs, p_values




def plot_gene_counts(adata: sc.AnnData, group_key: str, sample_key: str, patient_key: str, region_mask_keys: List[str], genes: str, save: Optional[str] =None, plot_horizontal: bool =True, axes: Optional[np.ndarray] =None, normalize: Optional[bool] = False):
    """
    Plot raw gene expression counts across tissue regions and groups.
    
    Creates boxplots showing mean gene expression (raw counts) per cell for 
    specified genes across different tissue regions, separated by two groups.
    Performs Mann-Whitney U tests and annotates significant differences.
    
    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object with raw counts in adata.raw.
    group_key : str
        Key in adata.obs containing exactly 2 group labels (must be categorical).
        Example: 'Growth pattern' with values ['EHGP', 'RHGP'].
    sample_key : str
        Key in adata.obs containing sample identifiers (must be categorical).
    patient_key : str
        Key in adata.obs containing patient identifiers (must be categorical).
        Expression values are averaged per patient to avoid pseudo-replication.
    region_mask_keys : List[str]
        List of boolean column names in adata.obs that define tissue regions.
        Each column should contain True/False indicating membership in that region.
        Example: ['Liver', 'Tumor', 'Stroma'].
    genes : List[str]
        List of gene names to plot. Gene names in italics on x-axis.
    save : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.
    plot_horizontal : bool, default=True
        Whether to plot the genes horizontally (True) or vertically (False).
    axes : Optional[np.ndarray], default=None
        Optional array of axes to plot on. If None, new figure and axes are created.
    normalize : bool, default=False
        Average number of reads per cell may vary due
        to overall sample quality. If True, mean expression of a gene in a region 
        is normalized by the mean total read count in that region.
        The normalized values are multiplied with the mean total read
        count across all samples and region to obtain pseudo-counts.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of subplot axes, one per region.
    
    Raises
    ------
    ValueError
        If adata.raw is None (raw counts required).
        If group_key, sample_key, or patient_key not in adata.obs.
        If any of these keys are not categorical dtype.
        If group_key does not have exactly 2 unique values.
    
    Notes
    -----
    - Expression values are averaged per sample, then per patient to avoid 
      pseudo-replication when comparing groups.
    - Statistical test: Mann-Whitney U test (two-sided).
    - Significance thresholds: * p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001.
    - Color scheme: '#6497bf' for first group, '#d8031c' for second group.
    - Y-axis shows mean raw count per cell.
    
    Examples
    --------
    >>> ad = sc.read_h5ad('data.h5ad')
    >>> fig, axs = plot_gene_counts(
    ...     ad,
    ...     group_key='Growth pattern',
    ...     sample_key='Sample',
    ...     patient_key='Patient',
    ...     region_mask_keys=['Tumor', 'Normal'],
    ...     genes=['CD8A', 'CD4', 'FOXP3'],
    ...     save='gene_expression.svg'
    ... )
    """

    # Try to get raw counts from adata
    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
    else:
        raise ValueError("adata must have raw counts in adata.raw to plot raw gene counts")
    
    # check that columns are categorical
    for key in [group_key, sample_key, patient_key]:
        if key not in adata.obs:
            raise ValueError(f"{key} not found in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[key]):
            raise ValueError(f"{key} must be a categorical column in adata.obs")
        
    unique_groups = adata.obs[group_key].cat.categories
    unique_samples = adata.obs[sample_key].cat.categories
    unique_patients = adata.obs[patient_key].cat.categories
    if len(unique_groups) != 2:
        raise ValueError(f"Expected exactly 2 unique groups in {group_key}, but found {len(unique_groups)}: {unique_groups}")
        
    # Growth pattern configuration
    palette = {unique_groups[0]: '#6497bf', unique_groups[1]: '#d8031c'}
    

    if normalize:
        mean_gene_count = np.mean(adata_raw.X.sum(axis=1).A.flatten())

    # Build results dataframe
    records = []
    for region in region_mask_keys:
        # Filter cells by region
        adata_raw_by_region = adata_raw[adata_raw.obs[region] == True]

        # Group by sample and gene
        for sample in unique_samples:
            adata_raw_by_sample = adata_raw_by_region[adata_raw_by_region.obs[sample_key] == sample]

            if not len(adata_raw_by_sample):
                continue
        
            group = adata_raw_by_sample.obs[group_key].iloc[0] if len(adata_raw_by_sample) else 'Unknown'
            patient = adata_raw_by_sample.obs[patient_key].iloc[0] if len(adata_raw_by_sample) else 'Unknown'

            if normalize:
                # Normalize the data by dividing with average total gene count per cell in the region
                # and multiply with average gene count per cell to obtain pseudo counts
                mean_gene_count_region = np.mean(adata_raw_by_sample.X.sum(axis=1).A.flatten())
                normalizer = mean_gene_count / mean_gene_count_region
            else:
                normalizer = 1.0

            for gene in genes:
                mean_expr = np.mean(adata_raw_by_sample[:, gene].X.toarray()) * normalizer
                records.append({
                    'Sample': sample,
                    'Patient': patient,
                    'Group': group,
                    'Gene': gene,
                    'Region': region,
                    'Average expression': mean_expr
                })
    
    df = pd.DataFrame(records)

    
    # Aggregate by patient and convert to categorical
    df = df.groupby(['Patient', 'Group', 'Gene', 'Region'])['Average expression'].mean().reset_index()
    df['Group'] = df['Group'].astype('category')
    df['Gene'] = df['Gene'].astype('category')


    # Plotting
    n_subplots = len(region_mask_keys)
    if axes is not None:
        if len(axes) != n_subplots:
            raise ValueError(f"Number of provided axes ({len(axes)}) does not match number of regions ({n_subplots})")
        fig = axes[0].figure
        axs = axes
    else:
        if plot_horizontal:
            height = BOXPLOT_HEIGHT
            width = BOXPLOT_WIDTH(len(genes)) * n_subplots
            fig, axs = plt.subplots(ncols=n_subplots, nrows=1, figsize=(width, height), sharex=False)
        else:
            height = BOXPLOT_HEIGHT * n_subplots
            width = BOXPLOT_WIDTH(len(genes))
            print(height, width)
            fig, axs = plt.subplots(ncols=1, nrows=n_subplots, figsize=(width, height), sharex=True)

    if n_subplots == 1:
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])  # ensure axs is always an array for consistent indexing

    # Plot each region
    for idx, region in enumerate(region_mask_keys):
        region_data = df.query('Region == @region')
        
        # Define statistical pairs: (gene A, EHGP) vs (gene A, RHGP) for each gene
        pairs = [((g, unique_groups[0]), (g, unique_groups[1])) for g in genes]
        
        # Create boxplot
        sns.boxplot(
            data=region_data,
            y='Average expression',
            x='Gene',
            hue='Group',
            order=genes,
            palette=palette,
            ax=axs[idx]
        )
        sns.despine()
        
        # Add statistical annotations
        annotator = Annotator(
            axs[idx],
            pairs,
            data=region_data,
            hue='Group',
            y='Average expression',
            x='Gene',
            order=genes
        )
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annotator.apply_and_annotate()
        
        # Styling
        axs[idx].legend(frameon=False, loc='best', ncols=1, fontsize=BOXPLOT_FONTSIZE)
        axs[idx].set_xlabel(None)
        if plot_horizontal and idx == 0:
            axs[idx].set_ylabel('Mean count per cell', fontsize=BOXPLOT_FONTSIZE)
        else:
            axs[idx].set_ylabel(None)
            
        axs[idx].set_title(f'{region} region', fontsize=BOXPLOT_FONTSIZE)

        # Only three y ticks, rounded to integers
        yticks = np.linspace(0, np.max(axs[idx].get_ylim()), 3)
        axs[idx].set_yticks(yticks)

        
        axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
        
        # Genes are italic
        for label in axs[idx].get_xticklabels():
            label.set_fontstyle('italic')
            label.set_fontsize(BOXPLOT_FONTSIZE)
            label.set_rotation(90)
        
        for label in axs[idx].get_yticklabels():
            label.set_fontsize(BOXPLOT_FONTSIZE)

        if plot_horizontal or n_subplots == 1:
            force_same_axis_height(axs[idx])

    #if not plot_horizontal:
    #    for ax in axs:
    #        ax.set_box_aspect(1)

    if save is not None:
        fig.savefig(save)
    return fig, axs



def plot_nhood_heatmap(
    scale_to_zscore,      # {"radius = 30 um": {"zscore": df}, "k = 10": {"zscore": df}, ...}
    focal,                 # str
    neighbor_order=None,   # list[str] | None
    scale_order=None,      # list[str] | None
    sort_desc=True,        # True -> largest mean to smallest
    vlim=None,             # (vmin, vmax) | None (auto symmetric if None)
    ax=None,
    title=None,
    cbar=True,
    save=None
):
    """
    Plot neighborhood z-scores for one focal cell type across scales as a heatmap.

    The function extracts z-score vectors for `focal` from each scale-specific
    matrix, stacks them into a 2D array (rows=scales, columns=neighbors),
    optionally sorts neighbors by mean z-score across scales, and renders the
    result with a diverging color map.

    Parameters
    ----------
    scale_to_zscore : dict[str, pandas.DataFrame]
        Mapping from scale label (for example ``"radius = 30 um"`` or
        ``"k = 10"``) to a z-score DataFrame. The DataFrame is expected to have
        cell types on both index and columns.
    focal : str
        Focal cell type to extract interactions for.
    neighbor_order : list[str] | None, default=None
        Neighbor cell type order to use on the x-axis. If ``None``, inferred
        from the first DataFrame.
    scale_order : list[str] | None, default=None
        Scale order for the y-axis. If ``None``, uses the insertion order of
        ``scale_to_zscore``.
    sort_desc : bool, default=True
        If ``True``, sort neighbors by descending mean z-score across scales.
        If ``False``, sort ascending.
    vlim : tuple[float, float] | None, default=None
        Color limits ``(vmin, vmax)``. If ``None``, limits are set
        symmetrically around zero using a robust percentile estimate.
    ax : matplotlib.axes.Axes | None, default=None
        Axis to draw on. If ``None``, creates a new figure and axis.
    title : str | None, default=None
        Optional plot title.
    cbar : bool, default=True
        Whether to draw a colorbar.
    save : str | None, default=None
        Optional path to save the figure. If ``None``, figure is not saved.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the heatmap.
    im : matplotlib.image.AxesImage
        Heatmap image returned by ``imshow``.

    Raises
    ------
    ValueError
        If ``scale_to_zscore`` is empty.
    """

    items = list(scale_to_zscore.items())
    if not items:
        raise ValueError("scale_to_zscore is empty")

    first_df = items[0][1]

    # Determine default neighbor order from first DF
    if neighbor_order is None:
        if focal in first_df.index:
            neighbor_order = list(first_df.columns)
        else:
            neighbor_order = list(first_df.index)

    # Determine scale order (y-axis)
    if scale_order is None:
        scale_order = [k for k in scale_to_zscore.keys()]
    else:
        scale_order = [s for s in scale_order if s in scale_to_zscore]

    # Helper: extract z-score vector for one scale, aligned to neighbor_order
    def _extract_vector(zdf, neighbors):
        if focal in zdf.index:
            sub = zdf.reindex(index=[focal], columns=neighbors)
            y = sub.loc[focal].values.astype(float)
        else:
            sub = zdf.reindex(index=neighbors, columns=[focal])
            y = sub[focal].values.astype(float)
        return y

    # Build matrix H (rows=scales, cols=neighbors)
    rows = []
    for s in scale_order:
        zdf = scale_to_zscore[s]
        y = _extract_vector(zdf, neighbor_order)
        rows.append(y)
    H = np.vstack(rows)

    # Sort neighbors by mean across scales
    col_means = np.nanmean(H, axis=0)
    filler = -np.inf if sort_desc else np.inf  # push all-NaN to end
    safe_means = np.where(np.isfinite(col_means), col_means, filler)
    order = np.argsort(safe_means)
    if sort_desc:
        order = order[::-1]
    neighbor_order_sorted = [neighbor_order[i] for i in order]
    H = H[:, order]

    # Symmetric color limits around 0 (robust)
    if vlim is None:
        finite = np.isfinite(H)
        if finite.any():
            vmax = float(np.nanpercentile(np.abs(H[finite]), 99))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.nanmax(np.abs(H))) if np.isfinite(np.nanmax(np.abs(H))) else 1.0
        else:
            vmax = 1.0
        vmin, vmax = -vmax, vmax
    else:
        vmin, vmax = vlim

    # Plot
    if ax is None:
        fig_w = max(4, 0.35 * H.shape[1] + 2)
        fig_h = max(2.5, 0.45 * H.shape[0] + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure

    im = ax.imshow(H, aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax, interpolation="nearest")

    # Ticks/labels
    ax.set_yticks(np.arange(len(scale_order)))
    # Italic y-tick labels (scales)
    ax.set_yticklabels(scale_order, fontstyle='italic')

    ax.set_xticks(np.arange(len(neighbor_order_sorted)))
    ax.set_xticklabels(neighbor_order_sorted, rotation=90)

    #ax.set_ylabel("Scale")
    ax.set_xlabel("Neighboring cell types")
    if title:
        ax.set_title(title, fontsize=10)

    # Colorbar
    if cbar:
        cax = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cax.set_label("Z-score")

    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    return ax, im




REGION_ORDER = ["Liver front", "Capsule (liver side)", "Capsule (tumor side)", "Tumor front"]
COLORBAR_TITLE = "Rel. count per area\n(averaged across patients)"

def _build_plot_adata(counts_per_area: pd.DataFrame) -> sc.AnnData:
    # Average across patients for each label, then orient matrix as regions x labels.
    counts_by_label = counts_per_area.copy().groupby("Label").mean()
    region_by_label = counts_by_label.T.reindex(REGION_ORDER)

    adata = sc.AnnData(
        X=region_by_label.to_numpy(),
        obs=pd.DataFrame(index=region_by_label.index),
        var=pd.DataFrame(index=region_by_label.columns),
    )
    adata.obs["Region"] = pd.Categorical(
        adata.obs.index, categories=REGION_ORDER, ordered=True
    )
    return adata

def _infer_gene_groups(adata: sc.AnnData) -> dict[str, list[str]]:
    expr = np.asarray(adata.X)
    dominant_region_idx = expr.argmax(axis=0)
    dominant_scores = expr[dominant_region_idx, np.arange(adata.n_vars)]

    groups: dict[str, list[str]] = {}
    for region_idx in np.unique(dominant_region_idx):
        mask = dominant_region_idx == region_idx
        genes = adata.var_names[mask]
        scores = dominant_scores[mask]
        genes_sorted = genes[np.argsort(scores)[::-1]]
        groups[REGION_ORDER[region_idx]] = genes_sorted.tolist()
    return groups

def _set_xtick_style(ax, italic_xticklabels: bool) -> None:
    font_style = "italic" if italic_xticklabels else "normal"
    for label in ax.get_xticklabels():
        label.set_fontstyle(font_style)

def plot_counts_per_ehgp_region(
    counts_per_area: pd.DataFrame,
    gene_groups: Optional[dict] = None,
    italic_xticklabels: bool = True,
    figsize=None,
    split_by_group: bool = False,
    show_groups: bool = True,
    save: Optional[str] = None
):
    """
    Plot EHGP region-level relative counts as a matrix plot.

    The function expects a table of relative counts per area where rows are
    indexed by ``("Patient", "Label")`` and columns correspond to region
    names. It averages values across patients for each label, reorders regions
    to a fixed display order, and renders a Scanpy matrix plot.

    Parameters
    ----------
    counts_per_area : pd.DataFrame
        Relative counts-per-area table with a MultiIndex that includes
        ``Label`` and region columns.
    gene_groups : Optional[dict], default=None
        Mapping of group name to list of labels to plot. If ``None``, labels are
        grouped automatically by the region where each label has its highest
        average value.
    italic_xticklabels : bool, default=True
        If ``True``, x-axis label names are rendered in italic.
    figsize : tuple | None, default=None
        Figure size passed to the underlying matrix plot.
    split_by_group : bool, default=False
        If ``True``, creates one subplot per group in ``gene_groups``.
        If ``False``, all groups are shown in one matrix plot.
    show_groups : bool, default=True
        If ``False``, group structure is flattened and labels are plotted as a
        single list.
    save : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    None
        Displays the generated matrix plot(s).
    """
    bwr = plt.get_cmap("bwr")
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_bwr", bwr(np.linspace(0.5, 1, 5))
    )
    adata = _build_plot_adata(counts_per_area)

    if gene_groups is None:
        gene_groups = _infer_gene_groups(adata)

    if not show_groups and isinstance(gene_groups, dict):
        gene_groups = [gene for genes in gene_groups.values() for gene in genes]

    n_genes = (
        sum(len(val) for val in gene_groups.values())
        if isinstance(gene_groups, dict)
        else len(gene_groups)
    )

    if split_by_group:
        if not isinstance(gene_groups, dict):
            gene_groups = {"Genes": list(gene_groups)}

        n_groups = len(gene_groups)
        fig, axs = plt.subplots(n_groups, 1, figsize=figsize)
        axs = np.atleast_1d(axs)
        plt.subplots_adjust(hspace=1.5)

        for i, (group, genes) in enumerate(gene_groups.items()):
            fig_sub = sc.pl.matrixplot(
                adata,
                genes,
                groupby="Region",
                var_group_rotation=0,
                colorbar_title=COLORBAR_TITLE,
                cmap=custom_cmap,
                return_fig=True,
                ax=axs[i],
                show=True,
                figsize=(figsize[0], 1) if figsize is not None else (n_genes, 1),
            )
            main_ax = fig_sub.get_axes()["mainplot_ax"]
            main_ax.set_title(group, fontsize=10)
            axs[i] = main_ax
            _set_xtick_style( axs[i], italic_xticklabels)
    else:
        fig, ax = plt.subplots(figsize=figsize if figsize is not None else (n_genes, 1))
        sub_fig = sc.pl.matrixplot(
            adata,
            gene_groups,
            groupby="Region",
            var_group_rotation=15,
            colorbar_title=COLORBAR_TITLE,
            cmap=custom_cmap,
            return_fig=True,
            ax=ax,
            show=True,
            figsize=figsize if figsize is not None else (n_genes, 1),
        )
        ax = sub_fig.get_axes()["mainplot_ax"]
        _set_xtick_style(ax, italic_xticklabels)

    if save is not None:
        fig.savefig(save)