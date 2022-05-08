import anndata as ad
import scanpy as sc
import numpy as np

from .utils.graph import get_shortest_paths, get_total_path
from .utils.mcv import mcv_pcs
from .utils import vprint

from scipy.sparse.csgraph import shortest_path


def program_times(data, cluster_obs_key_dict, cluster_order_dict, layer="X", program_var_key='program',
                  programs=['0', '1'], verbose=False):

    if type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    for prog in programs:

        _obsk = f"program_{prog}_time"
        _obsmk = f"program_{prog}_pca"

        _var_idx = data.var[program_var_key] == prog

        vprint(f"Assigning time values for program {prog} "
               f"containing {np.sum(_var_idx)} genes",
               verbose=verbose)

        _cluster_labels = data.obs[cluster_obs_key_dict[prog]].values

        if np.sum(_var_idx) == 0:
            data.obs[_obsk] = np.nan

        else:
            lref = data.X if layer == "X" else data.layers[layer]

            data.obs[_obsk], data.obsm[_obsmk], data.uns[_obsmk] = _calculate_program_time(
                lref[:, _var_idx],
                _cluster_labels,
                cluster_order_dict[prog],
                return_components=True,
                verbose=verbose
            )

            data.uns[_obsmk]['obs_time_key'] = _obsk
            data.uns[_obsmk]['obs_group_key'] = cluster_obs_key_dict[prog]
            data.uns[_obsmk]['obsm_key'] = _obsmk
            data.uns[_obsmk]['cluster_order_dict'] = cluster_order_dict[prog]

    return data


def _calculate_program_time(count_data, cluster_vector, cluster_order_dict, n_neighbors=10,
                            n_comps=None, graph_method="D", return_components=False, verbose=False):

    n = count_data.shape[0]

    if not np.all(np.isin(
        np.array(list(cluster_order_dict.keys())),
        np.unique(cluster_vector))
    ):
        raise ValueError(
            f"Mismatch between cluster_order_dict keys {list(cluster_order_dict.keys())} "
            f"And cluster_vector values {np.unique(cluster_vector)}"
        )

    adata = ad.AnnData(count_data, dtype=float)
    sc.pp.normalize_per_cell(adata, min_counts=0)
    sc.pp.log1p(adata)

    if n_comps is None:
        n_comps = np.median(mcv_pcs(count_data, n=1), axis=0).argmin()

    sc.pp.pca(adata, n_comps=n_comps, zero_center=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)

    vprint(f"Preprocessed expression count data {count_data.shape}",
           verbose=verbose)

    centroids = {k: adata.obs_names.get_loc(adata.obs_names[cluster_vector == k][idx])
                 for k, idx in get_centroids(adata.obsm['X_pca'], cluster_vector).items()}

    centroid_indices = [centroids[k] for k in centroids.keys()]

    vprint(f"Identified centroids for groups {', '.join(centroids.keys())}",
           verbose=verbose)

    # Order the centroids and build an end-to-end path
    _total_path, _tp_centroids = get_total_path(
        get_shortest_paths(
            adata.obsp['distances'],
            centroid_indices,
            graph_method=graph_method
        ),
        cluster_order_dict,
        list(centroids.keys())
    )

    vprint(f"Built {len(_total_path)} length path connecting "
           f"{len(_tp_centroids)} groups",
           verbose=verbose)

    # Find the nearest points on the shortest path line for every point
    # As numeric position on _total_path
    _nearest_point_on_path = shortest_path(
        adata.obsp['distances'],
        directed=False,
        indices=_total_path[:-1],
        return_predecessors=False,
        method=graph_method
    ).argmin(axis=0)

    # Scalar projections onto centroid-centroid vector
    times = np.full(n, np.nan, dtype=float)

    group = {
        'index': np.zeros(n, dtype=int),
        'names': [],
        'centroids': [],
        'path': []
    }

    for start, (end, left_time, right_time) in cluster_order_dict.items():

        if _tp_centroids[end] == 0:
            _right_centroid = len(_total_path)
        else:
            _right_centroid = _tp_centroids[end]

        _idx = _nearest_point_on_path >= _tp_centroids[start]
        _idx &= _nearest_point_on_path < _right_centroid

        vprint(f"Assigned times to {np.sum(_idx)} cells [{start} - {end}] "
               f"conected by {_right_centroid - _tp_centroids[start]} points",
               verbose=verbose)

        times[_idx] = scalar_projection(
            adata.obsm['X_pca'][:, 0:2],
            centroids[start],
            centroids[end]
        )[_idx] * (right_time - left_time) + left_time

        group['index'][_idx] = len(group['names'])
        group['names'].append(f"{start} / {end}")
        group['centroids'].append((centroids[start], centroids[end]))
        group['path'].append(_total_path[_tp_centroids[start]:_right_centroid])

    if verbose:
        vprint(f"Assigned times to {np.sum(~np.isnan(times))} cells "
               f"[{np.nanmin(times):.3f} - {np.nanmax(times):.3f}]",
               verbose=verbose)

    adata.uns['pca']['centroids'] = centroids
    adata.uns['pca']['shortest_path'] = _total_path
    adata.uns['pca']['closest_path_assignment'] = group['index']
    adata.uns['pca']['assignment_names'] = group['names']
    adata.uns['pca']['assignment_centroids'] = group['centroids']
    adata.uns['pca']['assignment_path'] = group['path']

    if return_components:
        return times, adata.obsm['X_pca'], adata.uns['pca']
    else:
        return times


def scalar_projection(data, center_point, off_point, normalize=True):

    vec = data[off_point, :] - data[center_point, :]

    scalar_proj = np.dot(
        data - data[center_point, :],
        vec
    )

    scalar_proj = scalar_proj / np.linalg.norm(vec)

    if normalize:
        _center_scale = scalar_proj[center_point]
        _off_scale = scalar_proj[off_point]
        scalar_proj = (scalar_proj - _center_scale) / \
            (_off_scale - _center_scale)

    return scalar_proj


def get_centroids(comps, cluster_vector):
    return {k: _get_centroid(comps[cluster_vector == k, :])
            for k in np.unique(cluster_vector)}


def _get_centroid(comps):
    return np.sum((comps - np.mean(comps, axis=0)[None, :]) ** 2, axis=1).argmin()
