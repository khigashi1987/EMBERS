import numpy as np
import umap
from sklearn.cluster import HDBSCAN

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def emb_2d_umap(ALL_EMBEDDINGS, n_neighbors, min_dist):
    model = umap.UMAP(verbose=True,
                      n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      n_components=2,
                      metric='cosine')
    result = model.fit_transform(ALL_EMBEDDINGS)
    return result

def run_hdbscan_clustering(coords_2d, min_cluster_size):
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    model.fit(coords_2d)
    return model.labels_