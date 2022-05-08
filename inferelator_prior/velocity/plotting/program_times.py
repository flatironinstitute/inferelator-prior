import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import pandas as pd
import numpy as np

DEFAULT_CMAP = 'plasma'


def program_time_summary(adata, program, cluster_order=None, cluster_colors=None, cbar_cmap=None, ax=None, hist_bins=80, cbar_title=None):

    # Get keys

    uns_key = f"program_{program}_pca"
    if uns_key not in adata.uns:
        raise ValueError(
            f"Unable to find program {program} in .uns[{uns_key}]. "
            "Run program_times() before calling plotter."
        )

    obs_group_key = adata.uns[uns_key]['obs_group_key']
    obs_time_key = adata.uns[uns_key]['obs_time_key']
    obsm_key = adata.uns[uns_key]['obsm_key']

    # Set up colormappings if not provided

    if cluster_order is None:
        cluster_order = np.unique(adata.obs[obs_group_key])

    _panels = adata.uns[uns_key]['assignment_names']
    n = len(_panels)

    if cbar_cmap is None:
        cbar_cmap = DEFAULT_CMAP

    cbar_cmap = cm.get_cmap(cbar_cmap, n)

    if cluster_colors is None:
        cluster_colors = {cluster_order[i]: colors.rgb2hex(cbar_cmap(i)) for i in range(n)}

    _color_vector = _get_colors(adata.obs[obs_group_key].values, cluster_colors)

    # Set up figure
    if ax is None:

        _layout = [['pca1'], ['pca2'], ['hist'], ['cbar']]

        if n < 5:
            _groups = [_panels[i] if i < n else ['.'] for i in range(4)]
            _layout = [[_layout[i] + [_groups[i]]] for i in range(4)]

        elif n < 9:
            _groups = [_panels[i] if i < n else ['.'] for i in range(8)]
            _layout = [[_layout[i] + [_groups[2*i], _groups[2*i + 1]]] for i in range(4)]

        else:
            _groups = [_panels[i] if i < n else ['.'] for i in range(12)]
            _layout = [[_layout[i] + [_groups[3*i], _groups[3*i + 1], _groups[3*i + 2]]]
                       for i in range(4)]

        fig, ax = plt.subplot_mosaic(_layout,
                                     gridspec_kw=dict(width_ratios=[1] * len(_layout[0]),
                                                      height_ratios=[1, 1, 1, 1],
                                                      wspace=0.25, hspace=0.25),
                                     figsize=(8, 8), dpi=300)

    refs = {}

    refs['pca1'] = _plot_pca(
        adata.obsm[obsm_key][:, 0:2],
        ax['pca1'],
        _color_vector,
        adata.uns[uns_key]['centroids'],
        adata.uns[uns_key]['shortest_path'],
    )

    ax['pca1'].set_title(f"Program {program} PCs")
    ax['pca1'].set_xlabel("PC1")
    ax['pca1'].set_ylabel("PC2")

    refs['pca2'] = _plot_pca(
        adata.obsm[obsm_key][:, [0, 2]],
        ax['pca2'],
        _color_vector,
    )

    ax['pca2'].set_xlabel("PC1")
    ax['pca2'].set_ylabel("PC3")

    for i, _pname in enumerate(_panels):
        _idx = adata.uns[uns_key]['closest_path_assignment'] == i
        refs['group'] = _plot_pca(
            adata.obsm[obsm_key][_idx, 0:2],
            ax[_pname],
            _color_vector[_idx],
            adata.uns[uns_key]['assignment_centroids'][i],
            adata.uns[uns_key]['assignment_path'][i]
        )

    refs['hist'] = _plot_time_histogram(
        adata.obs[obs_time_key].values,
        adata.obs[obs_group_key].values,
        ax['hist'],
        group_order=cluster_order,
        group_colors=cluster_colors,
        bins=hist_bins
    )

    _add_legend(
        ax['cbar'],
        cluster_colors,
        cluster_order,
        title=cbar_title
    )

    return fig, ax


def _plot_pca(comps, ax, colors, centroid_indices=None, shortest_path=None, s=1, alpha=0.5):

    _xlim = comps[:, 0].min(), comps[:, 0].max()
    _ylim = comps[:, 1].min(), comps[:, 1].max()

    rgen = np.random.default_rng(123)
    overplot_shuffle = np.arange(comps.shape[0])
    rgen.shuffle(overplot_shuffle)

    scatter_ref = ax.scatter(comps[overplot_shuffle, 0], comps[overplot_shuffle, 1],
                             c=colors[overplot_shuffle],
                             s=s, alpha=alpha)

    if centroid_indices is not None:
        ax.scatter(comps[centroid_indices, 0], comps[centroid_indices, 1],
                   c='None', edgecolor='black', s=150 * s, alpha=1)

        for i in range(len(centroid_indices) - 1):
            ax.plot(comps[[centroid_indices[i], centroid_indices[i + 1]], 0],
                    comps[[centroid_indices[i], centroid_indices[i + 1]], 1],
                    ls = '--', color='black',
                    alpha = 0.5,
                    lw = 1)

    if shortest_path is not None:
        ax.plot(comps[shortest_path, 0], comps[shortest_path, 1],
                '-ok', color='black', markersize=3, linewidth=1)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)

    return scatter_ref


def _get_colors(values, color_dict):

    c = np.empty_like(values, dtype=object)
    for k, col in color_dict.items():
        c[c == k] = col

    return c


def _get_time_hist_data(time_data, group_data, bins, group_order=None):

    if group_order is None:
        group_order = np.unique(group_data)

    cuts = np.linspace(np.min(time_data), np.max(time_data), bins)
    return [np.bincount(pd.cut(time_data[group_data == x],
                               cuts, labels=np.arange(len(cuts) - 1)).dropna(),
                        minlength=len(cuts) - 1) for x in group_order]


def _plot_time_histogram(time_data, group_data, ax, group_order=None, group_colors=None, bins=50):

    if group_order is None:
        group_order = np.unique(group_data)

    if group_colors is not None:
         _cmap = cm.get_cmap(DEFAULT_CMAP, len(group_order))
         group_colors = [colors.rgb2hex(_cmap(i)) for i in range(len(group_order))]

    hist_limit, fref = [], []
    hist_labels = np.linspace(0.5, bins - 0.5, bins)

    bottom_line = None
    for j, hist_data in enumerate(_get_time_hist_data(time_data, group_data, bins, group_order)):

        bottom_line = np.zeros_like(hist_data) if bottom_line is None else bottom_line

        fref.append(ax.bar(hist_labels,
                           hist_data,
                           bottom=bottom_line,
                           width=0.5,
                           color=group_colors[j]))

        bottom_line = bottom_line + hist_data

        hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)
    ax.set_ylim(0, hist_limit)
    ax.set_xlim(0, bins)

    return fref


def _add_legend(ax, colors, labels, title=None, **kwargs):
    ax.axis('off')
    _ = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(frameon=False,
                     loc='center left',
                     ncol=1,
                     borderpad=0.1,
                     borderaxespad=0.1,
                     columnspacing=0,
                     mode=None,
                     title=title,
                     **kwargs)
