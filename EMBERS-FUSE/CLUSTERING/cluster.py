import json
import numpy as np
import umap
from sklearn.cluster import HDBSCAN
import leidenalg
import igraph
from sklearn.neighbors import NearestNeighbors
from collections import deque
from scipy.cluster.hierarchy import linkage,  to_tree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import logging

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

def run_leiden_clustering(coords_2d, n_neighbors=5, resolution=0.01):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, 
                            metric='cosine').fit(coords_2d)
    distances, indices = nbrs.kneighbors(coords_2d)

    edges = []
    edge_weights = []
    for i in range(coords_2d.shape[0]):
        for j in range(1, n_neighbors + 1):
            edges.append((i, indices[i][j]))
            edge_weights.append(1 - distances[i][j])

    g = igraph.Graph(edges=edges, directed=False)
    g.es['weight'] = edge_weights

    partition = leidenalg.find_partition(g, 
                                         leidenalg.CPMVertexPartition,
                                         weights='weight',
                                         resolution_parameter=resolution)
    return np.array(partition.membership)

def run_matching_keys(keys_embeddings, keys_texts, llm,
                      purity_threshold=0.8, min_size=10):
    logging.info('Running matching keys...')
    # Start hierarchical-clustering keys and convert linkage to a tree
    linkage_matrix = linkage(keys_embeddings, 
                             metric='cosine',
                             method='average')
    tree, nodelist = to_tree(linkage_matrix, rd=True)

    # Queue for BFS, initialized with the root node
    queue = deque([tree])
    pure_clusters = []

    while queue:
        node = queue.popleft()

        # If node is not a leaf, process it
        if not node.is_leaf():
            # Collect indices of all items under this node
            node_indices = []
            leaf_queue = deque([node])

            # Traverse to collect all leaf indices under the current node
            while leaf_queue:
                current_node = leaf_queue.popleft()
                if current_node.is_leaf():
                    node_indices.append(current_node.id)
                else:
                    leaf_queue.append(current_node.left)
                    leaf_queue.append(current_node.right)
            
            if len(node_indices) < min_size:
                # Skip if the size is too small
                continue

            # Calculate purity if the size condition is met
            current_texts = ["Key: "+keys_texts[i]['Key']+ \
                             "  Description: "+keys_texts[i]['Description'] + \
                             "  Examples: "+str(keys_texts[i]['Example_values'])
                                for i in node_indices]
            if len(current_texts) > 100:
                logging.info('\tDiversity-preserving sampling...')
                sampled_node_indices = sample_by_pca_clustering(keys_embeddings[node_indices, :], n_samples=100)
                query_texts = [current_texts[i] for i in sampled_node_indices]
                logging.info('\tDiversity-preserving sampling...Done.')
            else:
                query_texts = current_texts
            result = llm.calculate_purity(query_texts)
            result = json.loads(result)
            purity = float(result['Purity'])
            logging.info(f'\tPurity calculation result: {result}')
            
            if purity < purity_threshold:
                queue.append(node.left)
                queue.append(node.right)
            else:
                # If purity is high enough, do not explore further
                pure_clusters.append({'Indices':node_indices,
                                       'Texts': current_texts,
                                       'Purity': purity})
    
    logging.info('Running matching keys...Done.')
    return pure_clusters

def sample_by_pca_clustering(data, n_samples=100, n_components=50, n_clusters=10):
    """
    Perform sampling from a dataset by reducing its dimensionality using PCA followed by clustering with K-Means. 
    This method aims to preserve the diversity of the dataset by ensuring samples from various clusters in the PCA-reduced space.
    
    Parameters:
        data (np.array): The input dataset of shape (num_samples, num_features).
        n_samples (int): The number of samples to extract.
        n_components (int): The number of principal components to retain in the PCA.
        n_clusters (int): The number of clusters to form in the reduced dimensionality space.

    Returns:
        np.array: An array of sampled data points from the original dataset.
    
    This function standardizes the input data, applies PCA to reduce its dimensionality, performs K-Means clustering on the reduced data,
    and samples uniformly from each cluster to achieve a diverse set of data points from across the principal component space.
    """
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    data_transformed = pca.fit_transform(data_normalized)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data_transformed)

    sample_indices = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) < (n_samples // n_clusters):
            sample_indices.extend(cluster_indices)
        else:
            sample_indices.extend(np.random.choice(cluster_indices, n_samples // n_clusters, replace=False))

    # 必要ならば残りのサンプルを追加サンプリング
    additional_samples_needed = n_samples - len(sample_indices)
    if additional_samples_needed > 0:
        additional_indices = np.random.choice(range(data.shape[0]), additional_samples_needed, replace=False)
        sample_indices.extend(additional_indices)

    return np.array(sample_indices)
