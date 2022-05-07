import anndata as ad
import scanpy as sc
import numpy as np

from .mcv import mcv_pcs
from . import vprint

from scipy.sparse.csgraph import shortest_path
import itertools


def program_times(data, cluster_obs_key_dict, cluster_order_dict, layer="X", program_var_key='program',
                  programs=['0', '1'], verbose=False):

    if type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    for prog in programs:

        vprint("Assigning time values for program {prog} "
               "from {cluster_obs_key_dict[prog]} groups",
               verbose=verbose)

        _obsk = f"program_{prog}_time"
        _obsmk = f"program_{prog}_pca"

        _var_idx = data.var[program_var_key] == prog

        _cluster_labels = data.obs[cluster_obs_key_dict[prog]].values

        if np.sum(_var_idx) == 0:
            data.obs[_obsk] = np.nan

        else:
            lref = data.X if layer == "X" else data.layers[layer]

            data.obs[_obsk], data.obsm[_obsmk], data.uns[_obsmk] = _calculate_program_time(
                lref[:, _var_idx],
                _cluster_labels,
                cluster_order_dict[prog],
                return_components=True
            )

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

    centroids = {k: adata.obs_names.get_loc(adata.obs_names[cluster_vector == k][idx])
                 for k, idx in get_centroids(adata.obsm['X_pca'], cluster_vector).items()}

    centroid_ids = list(centroids.keys())
    centroid_indices = [centroids[k] for k in centroid_ids]

    # Order the centroids and build an end-to-end path
    _total_path, _tp_centroids = _get_total_path(
        _get_shortest_path(
            adata.obsp['distances'],
            centroid_indices,
            graph_method=graph_method
        ),
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


def _get_shortest_path(graph, select_nodes, graph_method="D"):
    """
    Find the pairwise shortest path between specific nodes in
    an undirected graph

    :param graph: N x N graph connecting N nodes
    :type graph: np.ndarray, sp.spmatrix
    :param select_nodes: Indices of nodes to connect
    :type select_nodes: np.ndarray, list
    :param graph_method: _description_, defaults to "D"
    :type graph_method: str, optional
    :return: Array of shortest-path lists
    :rtype: np.ndarray [list] [N x N]
    """

    # Get the predecessor array from all data points to each node
    _, graph_pred = shortest_path(
        graph,
        directed=False,
        indices=select_nodes,
        return_predecessors=True,
        method=graph_method
    )

    _n_nodes = len(select_nodes)

    # Build an N x N array of lists
    shortest_paths = np.zeros((_n_nodes, _n_nodes), dtype=object)

    # Add 1-len lists to the diagonal
    for i in range(_n_nodes):
        shortest_paths[i, i] = [select_nodes[i]]

    # For every combination of nodes
    for end_idx, start_idx in itertools.combinations(range(_n_nodes), 2):

        # Find the endpoint at the left node
        end = select_nodes[end_idx]

        # Find the start point at the right node
        current_loc = select_nodes[start_idx]
        path = [current_loc]

        # While the current position is different from the end position
        # Walk backwards through the predecessor array
        # Putting each location in the path list
        while current_loc != end:
            current_loc = graph_pred[end_idx, current_loc]
            path.append(current_loc)

        # Put the list in the output array in the correct position
        # And then reverse it for the other direction
        shortest_paths[end_idx, start_idx] = path[::-1]
        shortest_paths[start_idx, end_idx] = path

    return shortest_paths


def _get_total_path(shortest_paths, centroid_order_dict, centroid_order_list):
    """
    Take an array of shortest paths between key nodes and
    find the total path that connects all key nodes

    :param shortest_paths: [N x N] Array of lists, where the list is the
        shortest path connecting key nodes
    :type shortest_paths: np.ndarray [list] [N x N]
    :param centroid_order_dict: Dict keyed by node labels. Values are
        ('next_node_label', time_left_node, time_right_node)
    :type centroid_order_dict: dict
    :param centroid_order_list: Node labels for shortest_paths array
    :type centroid_order_list: np.ndarray, list
    :return: A list of nodes that connect every key node
        A dict keyed by key node label with the position of that label
        on the total path list
    :rtype: list, dict
    """

    total_path = []
    total_path_centroids = {}

    # Path goes from left to right node
    for i, (start_label, (end_label, _, _)) in enumerate(centroid_order_dict.items()):

        # Get the shortest path that starts at left and ends at right
        _link_path = shortest_paths[[start_label == x for x in centroid_order_list],
                                    [end_label == x for x in centroid_order_list]][0]

        # Set the position of the key node on the total path
        total_path_centroids[start_label] = len(total_path)
        total_path.extend(_link_path[int(i > 0):])

    return total_path, total_path_centroids
