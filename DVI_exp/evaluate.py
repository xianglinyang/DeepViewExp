from sklearn.neighbors import KDTree
from sklearn.manifold import trustworthiness
# from deepvisualinsight import backend
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from pynndescent import NNDescent
import numba

@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.
    Parameters
    ----------
    X: array of shape (n_samples, n_samples)
        The input data to compute the k-neighbor indices of.
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.
    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


def evaluate_proj_nn_perseverance_knn(dists, embedding, n_neighbors, metric="euclidean"):
    train_num = dists.shape[0]
    n_trees = 5 + int(round(train_num ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(train_num))))
    # get nearest neighbors
    high_ind = fast_knn_indices(dists, n_neighbors)
    nnd = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    low_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(train_num)
    for i in range(train_num):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))
    return border_pres.mean()


def evaluate_proj_boundary_perseverance_knn(dists, embedding, low_centers, n_neighbors):
    # dists[:train_num, train_num:]
    train_num = dists.shape[0]
    high_ind = fast_knn_indices(dists, n_neighbors)
    low_tree = KDTree(low_centers)
    _, low_ind = low_tree.query(embedding, k=n_neighbors)
    border_pres = np.zeros(train_num)
    for i in range(train_num):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))
    return border_pres.mean()


def evaluate_inv_distance(data, inv_data):
    return np.linalg.norm(data-inv_data, axis=1).mean()


def evaluate_inv_accu(labels, pred):
    return np.sum(labels == pred) / len(labels)


def evaluate_inv_conf(labels, ori_pred, new_pred):
    old_conf = [ori_pred[i, labels[i]] for i in range(len(labels))]
    new_conf = [new_pred[i, labels[i]] for i in range(len(labels))]
    old_conf = np.array(old_conf)
    new_conf = np.array(new_conf)

    diff = np.abs(old_conf - new_conf)
    # return diff.mean(), diff.max(), diff.min()
    return diff.mean()


def find_neighbor_preserving_rate(prev_distance, curr_distance, n_neighbors):
    """
    neighbor preserving rate, (0, 1)
    :param prev_data: ndarray, shape(N,2) low dimensional embedding from last epoch
    :param train_data: ndarray, shape(N,2) low dimensional embedding from current epoch
    :param n_neighbors:
    :return alpha: ndarray, shape (N,)
    """
    if prev_distance is None:
        return np.zeros(len(curr_distance))
    prev_ind = fast_knn_indices(prev_distance, n_neighbors)
    curr_ind = fast_knn_indices(curr_distance, n_neighbors)
    temporal_pres = np.zeros(len(curr_distance))
    for i in range(len(prev_ind)):
        pres = np.intersect1d(prev_ind[i], curr_ind[i])
        temporal_pres[i] = len(pres) / float(n_neighbors)
    return temporal_pres


def evaluate_proj_temporal_perseverance_corr(alpha, delta_x):
    """
    Evaluate temporal preserving property,
    calculate the correlation between neighbor preserving rate and moving distance in low dim in a time sequence
    :param alpha: ndarray, shape(N,) neighbor preserving rate
    :param delta_x: ndarray, shape(N,), moved distance in low dim for each point
    :return corr: ndarray, shape(N,), correlation for each point from temporal point of view
    """
    alpha = alpha.T
    delta_x = delta_x.T
    shape = alpha.shape
    data_num = shape[0]
    corr = np.zeros(data_num)
    for i in range(data_num):
        # correlation, pvalue = spearmanr(alpha[:, i], delta_x[:, i])
        correlation, pvalue = pearsonr(alpha[i], delta_x[i])
        if np.isnan(correlation):
            correlation = 0.0
        corr[i] = correlation
    return corr.mean()