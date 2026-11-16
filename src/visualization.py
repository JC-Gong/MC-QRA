import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import gaussian_kde
import libpysal
from esda.getisord import G_Local
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.text import Text
from shapely.geometry import LineString
import yaml
import os

_DEFAULT_CARTO = dict(
    projection="EPSG:4326",       
    ocean_color="#cfe3f2",
    land_color="#ece8dc",
    land_edge="#a9a99a",          
    graticule_step=30,            
    graticule_color="#b9b9b9",
    show_scale_bar=True,
    show_north_arrow=True,
    show_neatline=True,
    scale_bar_km=2000,            
    kde_classes=9,                
    tick_fontsize=15,             
    axis_fontsize=15,             
    scalebar_fontsize=15,         
    north_fontsize=15,            
    legend_fontsize=15,           
)


def _ensure_crs(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _build_graticule(step, target):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    segs = []
    for lon in range(-180, 181, step):
        xs, ys = [], []
        for lat in range(-90, 91, 5):
            x, y = t.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        segs.append(LineString([(xs[0], ys[0]), (xs[-1], ys[-1])]))
    for lat in range(-60, 61, step):
        xs, ys = [], []
        for lon in range(-180, 181, 5):
            x, y = t.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        segs.append(LineString([(xs[0], ys[0]), (xs[-1], ys[-1])]))
    return gpd.GeoDataFrame(geometry=segs, crs=target)


class AcademicPlotter:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            self.viz_config = full_config['visualization']
            self.output_dir = full_config['data'].get('output_dir', 'output')
        self._last_kde_cf = None
        self._setup_style()

    def _setup_style(self):
        """"""
        style = self.viz_config['style']
        plt.rcParams['font.family'] = style['font_family']
        plt.rcParams['font.size'] = style['font_size']
        plt.rcParams['axes.labelweight'] = style['font_weight']
        plt.rcParams['axes.titleweight'] = style['font_weight']

    def _apply_axes_style(self, ax):
        """"""
        style = self.viz_config['style']
        if style.get('remove_spines'):
            for spine in style['remove_spines']:
                ax.spines[spine].set_visible(False)
        if style.get('remove_title'):
            ax.set_title("")

    def _get_save_path(self, filename):
        """"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return os.path.join(self.output_dir, filename)

    def _carto(self):
        cfg = self.viz_config.get('cartography', {}) or {}
        d = dict(_DEFAULT_CARTO)
        d.update(cfg)
        return d

    def _project(self, gdf, world):
        target = self._carto()['projection']
        world = _ensure_crs(world)
        gdf = _ensure_crs(gdf)
        return gdf.to_crs(target), world.to_crs(target), target

    

    def _draw_basemap(self, ax, world_proj, target, extent, carto, with_carto=True,
                      scale_bar_corner='lower left', lon_step=None, show_lon_label=True):
        """ ArcGIS :  +  +  + ; /"""
        from pyproj import Transformer
        x0, y0, x1, y1 = extent
        ax.set_facecolor(carto['ocean_color'])
        
        for s in ax.spines.values():
            s.set_visible(False)
        world_proj.plot(ax=ax, facecolor=carto['land_color'],
                        edgecolor=carto['land_edge'], linewidth=0.3, zorder=1)
        step = carto['graticule_step']
        grat = _build_graticule(step, target)
        grat.plot(ax=ax, color=carto['graticule_color'], linewidth=0.35, zorder=2)

        
        t = Transformer.from_crs("EPSG:4326", target, always_xy=True)
        xticks, xlabels, yticks, ylabels = [], [], [], []
        for lon in range(-180, 181, lon_step or step):
            xp, _ = t.transform(lon, 0)
            xticks.append(xp)
            xlabels.append(f"{lon}\u00b0")
        for lat in range(-60, 61, step):
            _, yp = t.transform(0, lat)
            yticks.append(yp)
            ylabels.append(f"{lat}\u00b0")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=carto['tick_fontsize'])
        if show_lon_label:
            ax.set_xlabel("Longitude", fontsize=carto['axis_fontsize'], fontweight='bold')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=carto['tick_fontsize'])
        ax.set_ylabel("Latitude", fontsize=carto['axis_fontsize'], fontweight='bold')
        ax.tick_params(length=0)

        if carto['show_neatline']:
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           edgecolor='black', linewidth=1.0, zorder=6))
        if with_carto:
            self._scale_bar(ax, target, extent, carto, corner=scale_bar_corner)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('auto')

    def _scale_bar(self, ax, target, extent, carto, corner='lower left'):
        from pyproj import Transformer
        x0, y0, x1, y1 = extent
        km = carto['scale_bar_km']
        
        
        t = Transformer.from_crs("EPSG:4326", target, always_xy=True)
        dlambda = km / 111.32
        xa, _ = t.transform(0.0, -40.0)
        xb, _ = t.transform(dlambda, -40.0)
        L_frac = abs(xb - xa) / (x1 - x0)
        bh = 0.012
        half = L_frac / 2.0
        if 'upper' in corner:
            x0f, y0f = 0.08, 0.95
            label_dy = -bh * 2.2
        else:
            x0f, y0f = 0.08, 0.03
            label_dy = bh * 1.6
        
        ax.add_patch(mpatches.Rectangle((x0f, y0f), half, bh, transform=ax.transAxes,
                                       facecolor='black', edgecolor='black', zorder=7))
        ax.add_patch(mpatches.Rectangle((x0f + half, y0f), half, bh, transform=ax.transAxes,
                                       facecolor='white', edgecolor='black', zorder=7))
        for f in (x0f, x0f + L_frac):
            ax.plot([f, f], [y0f, y0f + bh], transform=ax.transAxes,
                    color='black', lw=0.8, zorder=7)
        ax.text(x0f + L_frac / 2, y0f + label_dy, f"{km:.0f} km", transform=ax.transAxes,
                ha='center', va='bottom' if label_dy > 0 else 'top',
                fontsize=carto['scalebar_fontsize'], fontweight='bold', zorder=7)

    def _density_levels(self, zz, n_classes):
        v = zz[np.isfinite(zz)]
        if len(v) < 2:
            return None
        try:
            import mapclassify as mc
            breaks = mc.NaturalBreaks(v, k=n_classes).bins
        except Exception:
            breaks = np.quantile(v, np.linspace(0, 1, n_classes + 1)[1:-1])
        breaks = np.unique(np.clip(breaks, v.min() + 1e-12, v.max() - 1e-12))
        levels = sorted(set([0.0] + breaks.tolist() + [float(v.max())]))
        return levels

    

    def plot_kde_map(self, gdf, world, year_range, filename="KDE.png"):
        """ KDE  ( +  + )"""
        y_min, y_max = year_range
        gdf_p, world_p, target = self._project(gdf, world)
        carto = self._carto()
        extent = tuple(world_p.total_bounds)
        gridsize = self.viz_config['kde']['gridsize']
        cmap = self.viz_config['kde']['cmap']
        alpha = self.viz_config['kde']['alpha']

        _df = gdf_p[(gdf_p['Year'] >= y_min) & (gdf_p['Year'] <= y_max)].dropna(
            subset=['Lon', 'Lat', 'risk'])
        fig, ax = plt.subplots(figsize=(15, 7.5))
        self._draw_basemap(ax, world_p, target, extent, carto)

        if len(_df) >= 2:
            values = np.vstack([_df.geometry.x, _df.geometry.y])
            kde = gaussian_kde(values, weights=_df['risk'], bw_method='scott')
            xx, yy = np.mgrid[extent[0]:extent[2]:complex(gridsize),
                              extent[1]:extent[3]:complex(gridsize)]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            levels = self._density_levels(zz, carto['kde_classes'])
            if levels:
                cf = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap,
                                 alpha=min(alpha, 0.55), extend='max', zorder=3,
                                 antialiased=True)
                ax.contour(xx, yy, zz, levels=levels, colors='#666666',
                           linewidths=0.3, alpha=0.3, zorder=3)
            else:
                cf = ax.contourf(xx, yy, zz, cmap=cmap,
                                 alpha=min(alpha, 0.55), zorder=3)
            self._last_kde_cf = cf
            
            world_p.plot(ax=ax, facecolor='none', edgecolor='#999999',
                         linewidth=0.7, zorder=4)
            cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02,
                                label="Risk density (weighted KDE)")
            cbar.ax.tick_params(labelsize=carto['tick_fontsize'])
            cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                                     useMathText=True)
            cbar.set_label("Risk density (weighted KDE)", fontsize=carto['axis_fontsize'])

        ax.scatter(_df.geometry.x, _df.geometry.y, s=8, c='#16324f',
                   alpha=0.5, linewidths=0, zorder=5)
        plt.savefig(self._get_save_path(filename), dpi=300, bbox_inches='tight')
        return fig

    def plot_hotspot_map(self, gdf, world, year_range, filename="HotSpot.png"):
        """ Getis-Ord Gi*  ( + )"""
        y_min, y_max = year_range
        gdf_p, world_p, target = self._project(gdf, world)
        carto = self._carto()
        extent = tuple(world_p.total_bounds)
        k_neighbors = self.viz_config['hotspot']['k_neighbors']
        colors = {
            0: "#615f5f", -1: '#bdd7e7', -2: '#6baed6', -3: '#08519c',
            1: '#fcae91', 2: '#fb6a4a', 3: '#67000d'
        }

        data_period = gdf_p[(gdf_p['Year'] >= y_min) & (gdf_p['Year'] <= y_max)].copy()
        if len(data_period) < 10:  
            return None

        w = libpysal.weights.KNN.from_dataframe(
            data_period, k=min(k_neighbors, len(data_period) - 1))
        w.transform = 'R'
        g_local = G_Local(data_period['risk'].astype(float), w, transform='B', star=True)
        data_period['GiZ'] = g_local.z_sim
        data_period['p_value'] = g_local.p_norm

        bins = np.zeros_like(data_period['GiZ'])
        bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.01)] = 3
        bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.05) & (data_period['p_value'] >= 0.01)] = 2
        bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.1) & (data_period['p_value'] >= 0.05)] = 1
        bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.1) & (data_period['p_value'] >= 0.05)] = -1
        bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.05) & (data_period['p_value'] >= 0.01)] = -2
        bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.01)] = -3
        data_period['Gi_Bin'] = bins

        fig, ax = plt.subplots(figsize=(12, 7))
        self._draw_basemap(ax, world_p, target, extent, carto, scale_bar_corner='upper left')
        
        world_p.plot(ax=ax, facecolor='none', edgecolor='#999999', linewidth=0.7, zorder=3)

        for k, color in colors.items():
            subset = data_period[data_period['Gi_Bin'] == k]
            if len(subset) == 0:
                continue
            ax.scatter(subset.geometry.x, subset.geometry.y, color=color,
                       s=22 if k != 0 else 10, alpha=1 if k != 0 else 0.45, zorder=4)

        legend_elements = [
            mlines.Line2D([], [], color='#67000d', marker='o', linestyle='None', markersize=8, label='Hot Spot (99%)'),
            mlines.Line2D([], [], color='#fb6a4a', marker='o', linestyle='None', markersize=8, label='Hot Spot (95%)'),
            mlines.Line2D([], [], color='#fcae91', marker='o', linestyle='None', markersize=8, label='Hot Spot (90%)'),
            mlines.Line2D([], [], color='#615f5f', marker='o', linestyle='None', markersize=6, alpha=0.5, label='Not Significant'),
            mlines.Line2D([], [], color='#bdd7e7', marker='o', linestyle='None', markersize=8, label='Cold Spot (90%)'),
            mlines.Line2D([], [], color='#6baed6', marker='o', linestyle='None', markersize=8, label='Cold Spot (95%)'),
            mlines.Line2D([], [], color='#08519c', marker='o', linestyle='None', markersize=8, label='Cold Spot (99%)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', frameon=True,
                  prop={'size': carto['legend_fontsize']})
        plt.savefig(self._get_save_path(filename), dpi=300, bbox_inches='tight')
        return fig

    def plot_kde_time_evolution(self, gdf, world, year_bins, year_labels,
                                filename="KDE_4periods.png"):
        """ KDE  (, /)"""
        n = len(year_labels)
        gdf_p, world_p, target = self._project(gdf, world)
        carto = self._carto()
        extent = tuple(world_p.total_bounds)
        gridsize = self.viz_config['kde']['gridsize']
        cmap = self.viz_config['kde']['cmap']
        alpha = self.viz_config['kde']['alpha']

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3.5), sharey=True,
                                 constrained_layout=True)
        if n == 1:
            axes = [axes]

        common_levels = None
        for i, (ax, ymin, ymax, label) in enumerate(
                zip(axes, year_bins[:-1], year_bins[1:], year_labels)):
            _df = gdf_p[(gdf_p['Year'] >= ymin) & (gdf_p['Year'] < ymax)].dropna(
                subset=['Lon', 'Lat', 'risk'])
            self._draw_basemap(ax, world_p, target, extent, carto, with_carto=True,
                                lon_step=60)

            if len(_df) >= 2:
                values = np.vstack([_df.geometry.x, _df.geometry.y])
                kde = gaussian_kde(values, weights=_df['risk'], bw_method='scott')
                xx, yy = np.mgrid[extent[0]:extent[2]:complex(gridsize),
                                  extent[1]:extent[3]:complex(gridsize)]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                if common_levels is None:
                    common_levels = self._density_levels(zz, carto['kde_classes'])
                if common_levels:
                    cf = ax.contourf(xx, yy, zz, levels=common_levels, cmap=cmap,
                                     alpha=min(alpha, 0.55), extend='max', zorder=3,
                                     antialiased=True)
                    ax.contour(xx, yy, zz, levels=common_levels, colors='#666666',
                               linewidths=0.3, alpha=0.3, zorder=3)
                else:
                    cf = ax.contourf(xx, yy, zz, cmap=cmap,
                                     alpha=min(alpha, 0.55), zorder=3)
                self._last_kde_cf = cf
                
                world_p.plot(ax=ax, facecolor='none', edgecolor='#999999',
                             linewidth=0.7, zorder=4)

            ax.scatter(_df.geometry.x, _df.geometry.y, s=8, c='#16324f',
                        alpha=0.5, linewidths=0, zorder=5)
            
            ax.text(0.03, 0.95, f"n={len(_df)}", transform=ax.transAxes,
                    ha='left', va='top', fontsize=carto['legend_fontsize'],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=8)

        if common_levels is not None and self._last_kde_cf is not None:
            cbar = fig.colorbar(self._last_kde_cf, ax=axes, fraction=0.02, pad=0.005,
                                label="Risk density (weighted KDE)")
            cbar.ax.tick_params(labelsize=carto['tick_fontsize'])
            cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                                     useMathText=True)
            cbar.set_label("Risk density (weighted KDE)", fontsize=carto['axis_fontsize'])
        plt.savefig(self._get_save_path(filename), dpi=300, bbox_inches='tight')
        return fig

    def plot_hotspot_time_evolution(self, gdf, world, year_bins, year_labels,
                                    filename="GiStar_HotSpot_4periods.png"):
        """ Getis-Ord Gi*  (, /)"""
        n = len(year_labels)
        gdf_p, world_p, target = self._project(gdf, world)
        k_neighbors = self.viz_config['hotspot']['k_neighbors']
        carto = self._carto()
        extent = tuple(world_p.total_bounds)
        colors = {
            0: "#615f5f", -1: '#bdd7e7', -2: '#6baed6', -3: '#08519c',
            1: '#fcae91', 2: '#fb6a4a', 3: '#67000d'
        }

        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]

        for i, (ax, ymin, ymax, label) in enumerate(
                zip(axes, year_bins[:-1], year_bins[1:], year_labels)):
            data_period = gdf_p[(gdf_p['Year'] >= ymin) & (gdf_p['Year'] < ymax)].copy()
            self._draw_basemap(ax, world_p, target, extent, carto, with_carto=True,
                               scale_bar_corner='lower left', lon_step=60)
            
            world_p.plot(ax=ax, facecolor='none', edgecolor='#999999', linewidth=0.7, zorder=3)

            if len(data_period) >= 10:
                w = libpysal.weights.KNN.from_dataframe(
                    data_period, k=min(k_neighbors, len(data_period) - 1))
                w.transform = 'R'
                g_local = G_Local(data_period['risk'].astype(float), w,
                                  transform='B', star=True)
                data_period['GiZ'] = g_local.z_sim
                data_period['p_value'] = g_local.p_norm

                bins = np.zeros_like(data_period['GiZ'])
                bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.001)] = 3
                bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.05) & (data_period['p_value'] >= 0.001)] = 2
                bins[(data_period['GiZ'] > 0) & (data_period['p_value'] < 0.1) & (data_period['p_value'] >= 0.05)] = 1
                bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.1) & (data_period['p_value'] >= 0.05)] = -1
                bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.05) & (data_period['p_value'] >= 0.001)] = -2
                bins[(data_period['GiZ'] < 0) & (data_period['p_value'] < 0.001)] = -3
                data_period['Gi_Bin'] = bins

                for k, color in colors.items():
                    subset = data_period[data_period['Gi_Bin'] == k]
                    if len(subset) == 0:
                        continue
                    ax.scatter(subset.geometry.x, subset.geometry.y, color=color,
                               s=22 if k != 0 else 10, alpha=1 if k != 0 else 0.45, zorder=4)

            ax.text(0.03, 0.95, f"n={len(data_period)}", transform=ax.transAxes,
                    ha='left', va='top', fontsize=carto['legend_fontsize'],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=8)

        legend_elements = [
            mlines.Line2D([], [], color='#67000d', marker='o', linestyle='None', markersize=8, label='Hot Spot (99%)'),
            mlines.Line2D([], [], color='#fb6a4a', marker='o', linestyle='None', markersize=8, label='Hot Spot (95%)'),
            mlines.Line2D([], [], color='#fcae91', marker='o', linestyle='None', markersize=8, label='Hot Spot (90%)'),
            mlines.Line2D([], [], color='#615f5f', marker='o', linestyle='None', markersize=6, alpha=0.5, label='Not Significant'),
            mlines.Line2D([], [], color='#bdd7e7', marker='o', linestyle='None', markersize=8, label='Cold Spot (90%)'),
            mlines.Line2D([], [], color='#6baed6', marker='o', linestyle='None', markersize=8, label='Cold Spot (95%)'),
            mlines.Line2D([], [], color='#08519c', marker='o', linestyle='None', markersize=8, label='Cold Spot (99%)')
        ]
        plt.tight_layout(rect=[0, 0.26, 1, 1])
        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True,
                   prop={'size': carto['legend_fontsize']})
        plt.savefig(self._get_save_path(filename), dpi=300, bbox_inches='tight')
        return fig

    def plot_risk_temporal_heatmap(self, df, filename="risk-temporal.png"):
        """"""
        risk_year_bin = df.groupby(["Year", "Risk bin"], observed=False).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(risk_year_bin.T, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Risk Level")
        plt.xticks(rotation=45)
        self._apply_axes_style(ax)
        plt.tight_layout()
        plt.savefig(self._get_save_path(filename), dpi=300)
        return fig

    def plot_nmi_heatmap(self, nmi_matrix, filename="NMI.png"):
        """ (NMI) """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(nmi_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
        self._apply_axes_style(ax)
        plt.tight_layout()
        plt.savefig(self._get_save_path(filename), dpi=300)
        return fig

    def plot_mi_barh(self, mi_matrix, target="Risk bin", exclude=None, filename="MI_Bar.png"):
        """ ( Risk Level)  RIF  (MI) """
        mi_series = mi_matrix.loc[target].drop(target)
        if exclude:
            mi_series = mi_series.drop([e for e in exclude if e in mi_series.index])
        mi_series = mi_series.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(mi_series.index, mi_series.values,
                       color=self.viz_config['style']['color'])
        for bar, val in zip(bars, mi_series.values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=self.viz_config['style']['font_size'])
        ax.set_xlabel("Mutual Information (MI)")
        ax.set_ylabel("RIF")
        ax.margins(x=0.12)
        self._apply_axes_style(ax)
        plt.tight_layout()
        plt.savefig(self._get_save_path(filename), dpi=300)
        return fig
