
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

plt.rcParams['svg.fonttype'] = 'none'
Image.MAX_IMAGE_PIXELS = 10_000_000_000


class SlidePlotter:
    """Plot helper for overlaying regions and density maps on DAPI images."""

    def __init__(self, ax, mpp: float = 0.16):
        """Initialize a plotting helper bound to a matplotlib axis.

        Parameters
        ----------
        ax
            Matplotlib axis used for all drawing operations.
        mpp
            Micrometers per pixel used by the scale bar and sigma conversion.
        """
        self.mpp = mpp
        self.ax = ax
        self.legend_handles = []
        self.shape = None

    def plot_dapi(self, path: Path, vmin: int = 25, vmax: int = 200) -> None:
        """Plot a grayscale DAPI image, hide axis decorations, and add a scale bar."""
        image = Image.open(path)
        self.ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        scalebar = ScaleBar(self.mpp, "um", length_fraction=0.25, location="lower left")
        self.ax.add_artist(scalebar)
        self.shape = image.size

    def plot_polygons(self, gdf, linewidth: float = 1, fill: bool = True) -> None:
        """Plot region polygons grouped by region and subregion.

        Parameters
        ----------
        gdf
            GeoDataFrame that includes at least geometry and region columns.
        linewidth
            Polygon boundary width.
        fill
            If True, fill polygons; otherwise draw outlines only.
        """
        for region, gdf_by_region in gdf.groupby("Region"):
            for subregion, gdf_by_subregion in gdf_by_region.groupby("Subregion"):
                color = (
                    gdf_by_subregion["color"].iloc[0]
                    if "color" in gdf_by_subregion.columns
                    else "red"
                )

                if fill:
                    gdf_by_subregion.plot(
                        ax=self.ax,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.75,
                        linewidth=linewidth,
                    )
                    legend_handle = Patch(
                        facecolor=color,
                        edgecolor=color,
                        label=subregion if pd.notna(subregion) else region,
                    )
                else:
                    gdf_by_subregion.plot(
                        ax=self.ax,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=linewidth,
                    )
                    legend_handle = Line2D(
                        [],
                        [],
                        color=color,
                        label=subregion if pd.notna(subregion) else region,
                        linewidth=linewidth,
                    )

                self.legend_handles.append(legend_handle)

    def make_legend(self) -> None:
        """Add a legend for region/subregion overlays if handles are available."""
        if self.legend_handles:
            self.ax.legend(
                handles=self.legend_handles,
                loc="lower right",
                frameon=True,
                fancybox=False,
                edgecolor="none",
                framealpha=1,
                borderaxespad=0.0,
            )

    def _fast_gaussian_filter(self, xy: np.ndarray, sigma_pixels: float) -> np.ndarray:
        """Return a blurred density map by filtering on a downsampled grid.

        The algorithm downsamples coordinates so the effective sigma is approximately
        5 pixels, applies a Gaussian filter, and resizes back to image resolution.
        """
        from skimage.transform import resize

        if self.shape is None:
            raise ValueError("Image shape is not set. Call plot_dapi before plot_density.")
        if xy.size == 0:
            return np.zeros(self.shape[::-1], dtype=float)

        downsample = max(sigma_pixels / 5.0, 1.0)
        density_shape = tuple(max(1, round(d / downsample)) for d in self.shape[::-1])
        density = np.zeros(density_shape, dtype=float)

        xy_down = np.round(xy / downsample).astype(int)
        in_bounds = (
            (xy_down[:, 0] >= 0)
            & (xy_down[:, 0] < density.shape[1])
            & (xy_down[:, 1] >= 0)
            & (xy_down[:, 1] < density.shape[0])
        )
        xy_down = xy_down[in_bounds]
        density[xy_down[:, 1], xy_down[:, 0]] = 1.0

        gaussian_filter(
            density,
            sigma=sigma_pixels / downsample,
            mode="constant",
            cval=0,
            output=density,
        )

        density = resize(
            density,
            self.shape[::-1],
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        return density

    def plot_density(
        self,
        xy: np.ndarray,
        sigma_um: float,
        percentile_clip: float = 97,
    ):
        """Overlay a clipped density map and add a compact horizontal colorbar.

        Parameters
        ----------
        xy
            N x 2 array of x/y coordinates in pixel space.
        sigma_um
            Gaussian smoothing sigma in micrometers.
        percentile_clip
            Upper percentile used for clipping and normalization.
        """
        bwr = plt.get_cmap("bwr")
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_bwr", bwr(np.linspace(0.5, 1, 5))
        )

        sigma_pixels = sigma_um / self.mpp
        density = self._fast_gaussian_filter(xy, sigma_pixels)

        vmax = np.percentile(density, percentile_clip) if density.size else 1.0
        vmax = max(vmax, 1e-12)
        vmax_str = f"P{int(percentile_clip)}"

        density = np.clip(density / vmax, 0, 1)
        density[0, 0] = 1.0
        alpha = np.clip(density.copy(), 0, 0.75)
        density_im = self.ax.imshow(
            density,
            cmap=custom_cmap,
            alpha=alpha,
            interpolation="nearest",
        )

        width = (
            "20%"
            if density.shape[1] > density.shape[0]
            else f"{max(1, round(density.shape[0] / density.shape[1] * 20))}%"
        )
        cax = inset_axes(
            self.ax,
            width=width,
            height="2%",
            loc="lower left",
            bbox_to_anchor=(0, -0.05, 1, 1),
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )

        cbar = Colorbar(ax=cax, mappable=density_im, orientation="horizontal")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", vmax_str])
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=8, direction="out", pad=1)
        cbar.set_label(f"Density [a.u]\nσ = {sigma_um} µm", fontsize=8, labelpad=1)

        return density_im
