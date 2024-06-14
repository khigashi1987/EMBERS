import os
import json
import pickle
import logging
import datetime
import numpy as np
from config import Config
from llm import LLM
from filemanager import FileManager
import cluster

class RunCluster():
    ###
    # Project, Methods, Keysの埋め込みをクラスタリングする
    ###
    def __init__(self, llm, filemanager):
        self.llm = llm
        self.files = filemanager
    
    def run_project(self, with_llm_summary=False):
        logging.info('Start clustering project')
        project_texts, project_embs = self.files.load_project()
        if project_texts is None or\
            project_embs is None:
            logging.error('No project embedding files')
            return
        logging.info(f'\tProject embeddings shape: {project_embs.shape}')
        
        logging.info('\tNormalize and reduce dimensionality of project embeddings...')
        project_embs = cluster.normalize_l2(project_embs[:, :Config.EMB_DIM_PROJECT])
        coords_2d = cluster.emb_2d_umap(project_embs, 
                                        n_neighbors=Config.N_NEIGHBORS_PROJECT, 
                                        min_dist=Config.MIN_DIST_PROJECT)
        np.save(os.path.join(Config.OUT_DIR, 'project_coords.npy'), coords_2d)
        logging.info('\tNormalize and reduce dimensionality of project embeddings...Done')
        
        logging.info('\tClustering project embeddings...')
        cluster_indices = cluster.run_hdbscan_clustering(coords_2d,
                                                         min_cluster_size=Config.MIN_CLUSTER_SIZE_PROJECT)
        np.save(os.path.join(Config.OUT_DIR, 'project_cluster_indices.npy'), cluster_indices)
        logging.info(f'\t\tNumber of clusters: {cluster_indices.max()+1}')
        logging.info('\tClustering project embeddings...Done')

        if with_llm_summary:
            logging.info('\tSummarizing clustering results...')
            project_labels = ['Unlabelled'] * project_embs.shape[0]
            for cluster_id in range(cluster_indices.max()+1):
                project_indices = np.where(cluster_indices == cluster_id)[0]
                project_texts_cluster = [project_texts[i]['key_findings'] for i in project_indices]
                result = self.llm.summarize_project_findings(project_texts_cluster)
                result = json.loads(result)
                for idx in project_indices:
                    project_labels[idx] = result['Topic']
                logging.info(f'\tCluster{cluster_id}: {result}')
            with open(os.path.join(Config.OUT_DIR, 'project_labels.pkl'), 'wb') as f:
                pickle.dump(project_labels, f)
            logging.info('\tSummarizing clustering results...Done')
        else:
            project_labels = ['Unlabelled'] * project_embs.shape[0]
            for i, cid in enumerate(cluster_indices):
                if cid != -1:
                    project_labels[i] = f'Cluster{cid}'
            with open(os.path.join(Config.OUT_DIR, 'project_labels.pkl'), 'wb') as f:
                pickle.dump(project_labels, f)

        logging.info('End clustering project')

    def run_methods(self, with_llm_summary=False):
        logging.info('Start clustering methods')
        methods_texts, methods_embs, methods_indices = self.files.load_methods()
        if methods_texts is None or\
            methods_embs is None or\
                methods_indices is None:
            logging.error('No methods embedding files')
            return
        logging.info(f'\tMethods embeddings shape: {methods_embs.shape}')

        logging.info('\tNormalize and reduce dimensionality of methods embeddings...')
        methods_embs = cluster.normalize_l2(methods_embs[:, :Config.EMB_DIM_METHODS])
        coords_2d = cluster.emb_2d_umap(methods_embs,
                                        n_neighbors=Config.N_NEIGHBORS_METHODS,
                                        min_dist=Config.MIN_DIST_METHODS)
        np.save(os.path.join(Config.OUT_DIR, 'methods_coords.npy'), coords_2d)
        logging.info('\tNormalize and reduce dimensionality of methods embeddings...Done')

        logging.info('\tClustering methods embeddings...')
        cluster_indices = cluster.run_hdbscan_clustering(coords_2d,
                                                         min_cluster_size=Config.MIN_CLUSTER_SIZE_METHODS)
        np.save(os.path.join(Config.OUT_DIR, 'methods_cluster_indices.npy'), cluster_indices)
        logging.info(f'\t\tNumber of clusters: {cluster_indices.max()+1}')
        logging.info('\tClustering methods embeddings...Done')

        if with_llm_summary:
            logging.info('\tSummarizing clustering results...')
            methods_labels = ['Unlabelled'] * methods_embs.shape[0]
            for cluster_id in range(cluster_indices.max()+1):
                methods_indices_cluster = np.where(cluster_indices == cluster_id)[0]
                methods_texts_cluster = [methods_texts[i] for i in methods_indices_cluster]
                # Using only 100 first texts for summarization
                result = self.llm.summarize_methods(methods_texts_cluster[:100])
                result = json.loads(result)
                for idx in methods_indices_cluster:
                    methods_labels[idx] = result['Label']
                logging.info(f'\tCluster{cluster_id}: {result}')
            with open(os.path.join(Config.OUT_DIR, 'methods_labels.pkl'), 'wb') as f:
                pickle.dump(methods_labels, f)
            logging.info('\tSummarizing clustering results...Done')
        else:
            methods_labels = ['Unlabelled'] * methods_embs.shape[0]
            for i, cid in enumerate(cluster_indices):
                if cid != -1:
                    methods_labels[i] = f'Cluster{cid}'
            with open(os.path.join(Config.OUT_DIR, 'methods_labels.pkl'), 'wb') as f:
                pickle.dump(methods_labels, f)

        logging.info('End clustering methods')

    def run_keys(self, with_llm_summary=False):
        logging.info('Start clustering keys')
        keys_texts, keys_embs = self.files.load_keys()
        if keys_texts is None or\
            keys_embs is None:
            logging.error('No keys embedding files')
            return
        logging.info(f'\tKeys embeddings shape: {keys_embs.shape}')

        logging.info('\tNormalize and reduce dimensionality of keys embeddings...')
        #keys_embs = cluster.normalize_l2(keys_embs[:, :Config.EMB_DIM_KEYS])
        coords_2d = cluster.emb_2d_umap(keys_embs,
                                        n_neighbors=Config.N_NEIGHBORS_KEYS,
                                        min_dist=Config.MIN_DIST_KEYS)  
        np.save(os.path.join(Config.OUT_DIR, 'keys_coords.npy'), coords_2d)
        logging.info('\tNormalize and reduce dimensionality of keys embeddings...Done')

        logging.info('\tClustering keys embeddings...')

        clusters_dict = cluster.run_matching_keys(keys_embs, keys_texts, self.llm,
                                                  purity_threshold=Config.KEYS_PURITY_THRESHOLD,
                                                  min_size=Config.KEYS_MIN_SIZE)

        cluster_indices = np.zeros(keys_embs.shape[0], dtype=int) - 1
        for i, cl in enumerate(clusters_dict):
            for idx in cl['Indices']:
                cluster_indices[idx] = i

        np.save(os.path.join(Config.OUT_DIR, 'keys_cluster_indices.npy'), cluster_indices)
        logging.info(f'\t\tNumber of clusters: {cluster_indices.max()+1}')
        logging.info('\tClustering keys embeddings...Done')

        if with_llm_summary:
            logging.info('\tSummarizing clustering results...')
            labels_descriptions = []
            keys_labels = ['Unlabelled'] * keys_embs.shape[0]
            for cluster_id in range(cluster_indices.max()+1):
                keys_indices_cluster = np.where(cluster_indices == cluster_id)[0]
                keys_texts_cluster =  clusters_dict[cluster_id]['Texts']
                result = self.llm.summarize_keys(keys_texts_cluster)
                result = json.loads(result)
                for idx in keys_indices_cluster:
                    keys_labels[idx] = result['Label']
                labels_descriptions.append(result)
                logging.info(f'\tCluster{cluster_id}: {result}')
            with open(os.path.join(Config.OUT_DIR, 'keys_labels.pkl'), 'wb') as f:
                pickle.dump(keys_labels, f)
            with open(os.path.join(Config.OUT_DIR, 'keys_labels_descriptions.json'), 'w') as f:
                json.dump(labels_descriptions, f)
            logging.info('\tSummarizing clustering results...Done')
        else:
            keys_labels = ['Unlabelled'] * keys_embs.shape[0]
            for i, cid in enumerate(cluster_indices):
                if cid != -1:
                    keys_labels[i] = f'Cluster{cid}'
            with open(os.path.join(Config.OUT_DIR, 'keys_labels.pkl'), 'wb') as f:
                pickle.dump(keys_labels, f)

        logging.info('End clustering keys')

if __name__ == '__main__':
    ### setup_logging()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(Config.LOG_DIR, f'log_{current_time}.txt'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    ###

    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ['project', 'methods', 'keys']:
        logging.error('Usage: python main.py project|methods|keys')
        sys.exit(1)

    llm = LLM(api_key=Config.OPENAI_API_KEY,
              completion_model_name=Config.COMPLETION_MODEL_NAME,
              max_tokens=Config.MAX_TOKENS)

    filemanager = FileManager(data_dir=Config.DATA_DIR, 
                              out_dir=Config.OUT_DIR)
    
    runcluster = RunCluster(llm=llm,
                            filemanager=filemanager)
    
    if sys.argv[1] == 'project':
        runcluster.run_project(with_llm_summary=True)
    elif sys.argv[1] == 'methods':
        runcluster.run_methods(with_llm_summary=True)
    elif sys.argv[1] == 'keys':
        runcluster.run_keys(with_llm_summary=True)
