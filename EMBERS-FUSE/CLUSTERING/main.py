import os
import json
import pickle
import logging
import datetime
import numpy as np
from config import Config
from llm import LLM
from filemanager import FileManager
from cluster import normalize_l2, \
                    emb_2d_umap, \
                    run_hdbscan_clustering

class RunCluster():
    ###
    # Project, Methods, Keysの埋め込みをクラスタリングする
    ###
    def __init__(self, llm, filemanager):
        self.llm = llm
        self.files = filemanager
    
    def run_project(self):
        logging.info('Start clustering project')
        project_texts, project_embs = self.files.load_project()
        if project_texts is None or\
            project_embs is None:
            logging.error('No project embedding files')
            return
        
        logging.info('\tNormalize and reduce dimensionality of project embeddings...')
        project_embs = normalize_l2(project_embs[:, :Config.EMB_DIM_PROJECT])
        coords_2d = emb_2d_umap(project_embs, 
                                n_neighbors=Config.N_NEIGHBORS, 
                                min_dist=Config.MIN_DIST)
        np.save(os.path.join(Config.OUT_DIR, 'project_coords.npy'), coords_2d)
        logging.info('\tNormalize and reduce dimensionality of project embeddings...Done')
        
        logging.info('\tClustering project embeddings...')
        cluster_indices = run_hdbscan_clustering(coords_2d, 
                                                 min_cluster_size=Config.PROJECT_MIN_CLUSTER_SIZE)
        np.save(os.path.join(Config.OUT_DIR, 'project_cluster_indices.npy'), cluster_indices)
        logging.info(f'\t\tNumber of clusters: {cluster_indices.max()+1}')
        logging.info('\tClustering project embeddings...Done')

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

        logging.info('End clustering project')

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
              model_name=Config.MODEL_NAME, 
              max_tokens=Config.MAX_TOKENS)
    
    filemanager = FileManager(data_dir=Config.DATA_DIR, 
                              out_dir=Config.OUT_DIR)
    
    cluster = RunCluster(llm=llm,
                         filemanager=filemanager)
    
    if sys.argv[1] == 'project':
        cluster.run_project()
    
