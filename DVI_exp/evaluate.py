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


def evaluate_proj_nn_perseverance_knn(dists, train_num, embedding, n_neighbors, metric="euclidean"):
    n_trees = 5 + int(round(train_num ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(train_num))))
    # get nearest neighbors
    high_ind = fast_knn_indices(dists[:train_num, :train_num])
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


def evaluate_proj_boundary_perseverance_knn(dists, train_num, embedding, low_centers, n_neighbors):
    high_ind = fast_knn_indices(dists[:train_num, train_num:], n_neighbors)
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

    diff = old_conf - new_conf
    # return diff.mean(), diff.max(), diff.min()
    return diff.mean()