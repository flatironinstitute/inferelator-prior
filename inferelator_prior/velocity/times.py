import anndata as ad
import scanpy as sc
import numpy as np

from .mcv import mcv_pcs
from scipy.sparse.csgraph import shortest_path
import itertools


def program_times(data, cluster_obs_key, cluster_order_dict, layer="X", program_var_key='program',
                  programs=['0', '1']):

    _cluster_labels = data.obs[cluster_obs_key].values

    if type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    for prog in programs:

        _obsk = f"program_{prog}_time"
        _obsmk = f"program_{prog}_pca"

        _var_idx = data.var[program_var_key] == prog

        if np.sum(_var_idx) == 0:
            data.obs[_obsk] = np.nan

        else:
            lref = data.X if layer == "X" else data.layers[layer]

            data.obs[_obsk], data.obsm[_obsmk], data.uns[_obsmk] = _calculate_program_time(
                lref[:, _var_idx],
                _cluster_labels,
                cluster_order_dict,
                return_components=True
            )

    return data


def _calculate_program_time(count_data, cluster_vector, cluster_order_dict, n_neighbors=10,
                            n_comps=None, graph_method="D", return_components=False):

    n = count_data.shape[0]

    adata = ad.AnnData(count_data, dtype=float)
    sc.pp.normalize_per_cell(adata, min_counts=0)
    sc.pp.log1p(adata)

    if n_comps is None:
        n_comps = np.median(mcv_pcs(count_data, n=1), axis=0).argmin()

    sc.pp.pca(adata, n_comps=n_comps, zero_center=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)

    centroids = {k: adata.obs_names.get_loc(adata.obs_names[cluster_vector == k][idx])
                 for k, idx in get_centroids(adata.obsm['X_pca'], cluster_vector).items()}

    centroid_ids = list(centroids.keys())
    centroid_indices = [centroids[k] for k in centroid_ids]

    # Get the shortest path between centroids from the graph
    _shortest_path = _get_shortest_path(
        adata.obsp['distances'],
        centroid_indices,
        graph_method=graph_method
    )

    # Order the centroids and build an end-to-end path
    _total_path, _tp_centroids = _get_total_path(
        _shortest_path,
        cluster_order_dict,
        centroid_ids
    )

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
    for left, (right, left_time, right_time) in cluster_order_dict.items():

        _right_centroid = _tp_centroids[right] if _tp_centroids[right] != 0 else len(
            _total_path)

        _idx = _nearest_point_on_path >= _tp_centroids[left]
        _idx &= _nearest_point_on_path < _right_centroid

        times[_idx] = scalar_projection(
            adata.obsm['X_pca'][:, 0:2],
            centroids[left],
            centroids[right]
        )[_idx] * (right_time - left_time) + left_time

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


def _get_shortest_path(graph, centroid_indices, graph_method="D"):
    # Find the shortest path between centroids
    graph_dist, graph_pred = shortest_path(
        graph,
        directed=False,
        indices=centroid_indices,
        return_predecessors=True,
        method=graph_method
    )

    n_centroids = len(centroid_indices)

    shortest_paths = np.zeros((n_centroids, n_centroids), dtype=object)

    for left, right in itertools.product(range(n_centroids), range(n_centroids)):
        start = centroid_indices[left]
        pred_arr = graph_pred[left, :]

        current_loc = centroid_indices[right]
        path = [current_loc]

        while current_loc != start:
            current_loc = pred_arr[current_loc]
            path.append(current_loc)

        shortest_paths[left, right] = path

    return shortest_paths


def _get_total_path(shortest_paths, centroid_order_dict, centroid_order_list):
    total_path = []
    total_path_centroids = {}

    for i, (left, (right, _, _)) in enumerate(centroid_order_dict.items()):
        _link_path = shortest_paths[[right == x for x in centroid_order_list],
                                    [left == x for x in centroid_order_list]][0]

        total_path_centroids[left] = len(total_path)
        total_path.extend(_link_path[int(i > 0):])

    return total_path, total_path_centroids
