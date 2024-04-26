import os
import glob
import pickle
import numpy as np
import logging

class FileManager():
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir

    def load_project(self):
        logging.info('Loading project embedding and text files...')
        project_embedding_file = os.path.join(self.out_dir, 'project_embedding.npy')
        project_texts_file = os.path.join(self.out_dir, 'project_texts.pkl')

        if os.path.exists(project_embedding_file) and \
            os.path.exists(project_texts_file):
            all_texts = pickle.load(open(project_texts_file, 'rb'))
            all_embeddings = np.load(project_embedding_file)
        else:
            all_texts = []
            all_embeddings = []
            for pkl_file in glob.glob(os.path.join(self.data_dir, 'PMC*/PMC*_project_embedding.pkl')):
                data = pickle.load(open(pkl_file, 'rb'))
                PMC_ID = os.path.basename(pkl_file).split('_')[0]
                key_findings = data['key_findings']
                all_texts.append({'PMC_ID': PMC_ID, 
                                  'key_findings': key_findings})
                vec = data['embedding']
                all_embeddings.append(vec)

            if len(all_embeddings) == 0:
                logging.error('No project embedding files')
                return None, None
            all_embeddings = np.vstack(all_embeddings)
            
            np.save(project_embedding_file, all_embeddings)
            with open(project_texts_file, 'wb') as f:
                pickle.dump(all_texts, f)
            
        logging.info('Loading project embedding and text files...Done')
        logging.info(f'\tProject embeddings shape: {all_embeddings.shape}')
        return all_texts, all_embeddings