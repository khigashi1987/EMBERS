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
    
    def load_methods(self):
        logging.info('Loading methods embedding and text files...')
        methods_embedding_file = os.path.join(self.out_dir, 'methods_embedding.npy')
        methods_indices_file = os.path.join(self.out_dir, 'methods_indices.pkl')
        methods_texts_file = os.path.join(self.out_dir, 'methods_texts.pkl')

        if os.path.exists(methods_embedding_file) and \
            os.path.exists(methods_indices_file) and \
                os.path.exists(methods_texts_file):
            all_methods_indices = pickle.load(open(methods_indices_file, 'rb'))
            all_texts = pickle.load(open(methods_texts_file, 'rb'))
            all_embeddings = np.load(methods_embedding_file)
        else:
            all_methods_indices = {}
            all_texts = []
            all_embeddings = []
            emb_indices = 0
            for pkl_file in glob.glob(os.path.join(self.data_dir, 'PMC*/PMC*_methods_embedding.pkl')):
                data = pickle.load(open(pkl_file, 'rb'))
                PMC_ID = os.path.basename(pkl_file).split('_')[0]
                all_methods_indices[PMC_ID] = {}
                if len(data['Sampling']) != 0:
                    vec1 = data['Sampling_embedding']
                    all_methods_indices[PMC_ID]['Sampling'] = [emb_indices, emb_indices + 1]
                    emb_indices += 1
                    all_embeddings.append(vec1)
                    all_texts.append(f'{PMC_ID} Sampling: '+data['Sampling'])
                
                if len(data['DNAExtraction']) != 0:
                    vecs2 = data['DNAExtraction_embedding']
                    if vecs2.ndim == 1:
                        len_vecs2 = 1
                    else:
                        len_vecs2 = vecs2.shape[0]
                    all_methods_indices[PMC_ID]['DNAExtraction'] = [emb_indices, emb_indices + len_vecs2]
                    emb_indices += len_vecs2
                    all_embeddings.append(vecs2)
                    all_texts.extend([f'{PMC_ID} DNA Extraction: '+de for de in data['DNAExtraction']])

            if len(all_embeddings) == 0:
                logging.error('No methods embedding files')
                return None, None, None
            all_embeddings = np.vstack(all_embeddings)
            
            np.save(methods_embedding_file, all_embeddings)
            with open(methods_indices_file, 'wb') as f:
                pickle.dump(all_methods_indices, f)
            with open(methods_texts_file, 'wb') as f:
                pickle.dump(all_texts, f)

        logging.info('Loading methods embedding and text files...Done')
        return all_texts, all_embeddings, all_methods_indices
