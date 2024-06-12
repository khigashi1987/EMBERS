import os
import glob
import pickle
import logging
from config import Config
from llm import LLM
import json
import datetime
import random

class Encoder():
    ###
    # PMCのディレクトリにある
    # PMC*_project.json, PMC*_methods.json, PMC*_new_keys_descriptions.jsonを読み込み、
    # OpenAI APIによる文書埋め込みをおこなう
    ###
    def __init__(self, llm=None, llm_for_keys=None):
        self.llm = llm
        self.llm_for_keys = llm_for_keys

    def encode_project(self, out_prefix):
        project_embedding_file  = f'{out_prefix}_project_embedding.pkl'
        project_json_file = f'{out_prefix}_project.json'

        if os.path.exists(project_embedding_file):
            logging.info(f'\tAlready analyzed project json file: {project_json_file}')
            return True
        
        if not os.path.exists(project_json_file):
            logging.info(f'\tNo project json file: {project_json_file}')
            return False
        
        with open(project_json_file, 'r') as f:
            project = json.load(f)
        project_text = project['key_findings']
        if len(project_text) == 0:
            vec = None
        else:
            vec = self.llm.get_embedding(project_text)
        project_embedding = {'key_findings': project_text, 'embedding': vec}
        with open(project_embedding_file, 'wb') as f:
            pickle.dump(project_embedding, f)
        
        return True
    
    def encode_methods(self, out_prefix):
        methods_embedding_file  = f'{out_prefix}_methods_embedding.pkl'
        methods_json_file = f'{out_prefix}_methods.json'

        if os.path.exists(methods_embedding_file):
            logging.info(f'\tAlready analyzed methods json file: {methods_json_file}')
            return True

        if not os.path.exists(methods_json_file):
            logging.info(f'\tNo methods json file: {methods_json_file}')
            return False
        
        with open(methods_json_file, 'r') as f:
            methods = json.load(f)
        methods_Sampling_text = methods['Sampling']
        if type(methods_Sampling_text) == list:
            methods_Sampling_text = ' '.join(methods_Sampling_text)
        if type(methods_Sampling_text) == dict:
            methods_Sampling_text = ' '.join(methods_Sampling_text.values())
        if len(methods_Sampling_text) == 0:
            sampling_vec = None
        else:
            sampling_vec = self.llm.get_embedding(methods_Sampling_text)

        methods_DNAExtraction_texts = methods['DNA extraction']
        if type(methods_DNAExtraction_texts) is list:
            methods_DNAExtraction_texts = [text for text in methods_DNAExtraction_texts if len(text) > 0]
        else:
            methods_DNAExtraction_texts = [str(methods['DNA extraction'])]
        if len(methods_DNAExtraction_texts) == 0:
            DNAExtraction_vec = None
        else:
            DNAExtraction_vec = self.llm.get_multiple_embedding(methods_DNAExtraction_texts)
        methods_embedding = {'Sampling': methods_Sampling_text, 
                             'DNAExtraction': methods_DNAExtraction_texts, 
                             'Sampling_embedding': sampling_vec, 
                             'DNAExtraction_embedding': DNAExtraction_vec}
        
        with open(methods_embedding_file, 'wb') as f:
            pickle.dump(methods_embedding, f)

        return True
    
    def encode_new_keys_descriptions(self, out_prefix):
        new_keys_descriptions_embedding_file  = f'{out_prefix}_new_keys_descriptions_embedding.pkl'
        new_keys_descriptions_json_file = f'{out_prefix}_new_keys_descriptions.json'
        samples_json_file = f'{out_prefix}_samples_update.json'

        if os.path.exists(new_keys_descriptions_embedding_file):
            logging.info(f'\tAlready analyzed new_keys_descriptions pickle file: {new_keys_descriptions_json_file}')
            return True

        if not os.path.exists(new_keys_descriptions_json_file):
            logging.info(f'\tNo new_keys_descriptions json file: {new_keys_descriptions_json_file}')
            return False
        
        with open(new_keys_descriptions_json_file, 'r') as f:
            new_keys_descriptions = json.load(f)
        
        with open(samples_json_file, 'r') as f:
            samples = json.load(f)
        
        new_keys_descriptions_embedding = []
        for k,v in new_keys_descriptions.items():
            if len(k) > 0 and len(v) > 0:
                target_string = f'{k}: {v}'

                # make "value-examples" from samples
                sample_values = [s[k] for s in samples if k in s and \
                                                          s[k] not in ['', 'NA', 'NaN', None] and \
                                                          type(s[k]) not in [list, dict]]
                # if the number of unique values is less than 5, use all unique values
                # otherwise, use 5 random unique values
                if len(set(sample_values)) <= 5:
                    example_values = list(set(sample_values))
                else:
                    example_values = random.sample(list(set(sample_values)), 5)

                vec = self.llm_for_keys.get_embedding(target_string)
                if vec is not None:
                    result = {'Key':k,
                              'Description':v,
                              'Example_values':example_values,
                              'Embedding':vec}
                    new_keys_descriptions_embedding.append(result)

        with open(new_keys_descriptions_embedding_file, 'wb') as f:
            pickle.dump(new_keys_descriptions_embedding, f)
        
        return True


    def encode(self, out_prefix):
        # Project embedding
        self.encode_project(out_prefix)
        
        # Method embedding
        self.encode_methods(out_prefix)

        # New keys embedding
        self.encode_new_keys_descriptions(out_prefix)
        
        return

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

    logging.info('Start analyzing process...')

    # Validation dataset
    TARGET_PMCs = [os.path.basename(pmcdir) for pmcdir in glob.glob(os.path.join(Config.RESULT_BASE_DIR, 'PMC*'))]

    for i, TARGET_PMC in enumerate(TARGET_PMCs):

        logging.info(f'{i} Analyzing PMC: {TARGET_PMC}')

        # Directories
        pmc_dir = os.path.join(Config.RESULT_BASE_DIR, f'{TARGET_PMC}')
        result_dir = os.path.join(Config.RESULT_BASE_DIR, f'{TARGET_PMC}')

        out_prefix = os.path.join(result_dir, f'{TARGET_PMC}')

        # Initialize
        llm = LLM(api_key=Config.OPENAI_API_KEY, model_name=Config.MODEL_NAME)
        llm_for_keys = LLM(api_key=Config.OPENAI_API_KEY, model_name=Config.MODEL_NAME_FOR_KEYS)
        encoder = Encoder(llm=llm, llm_for_keys=llm_for_keys)

        # Encode all
        encoder.encode(out_prefix)

        logging.info(f'End Analyzing PMC: {TARGET_PMC}\n\n')

    logging.info('End analyzing process.')
