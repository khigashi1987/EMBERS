import os
import numpy as np
import json
import logging
from tqdm import tqdm
from config import Config
from llm import LLM
import datetime
from filemanager import FileManager

class Aligner():
    def __init__(self, llm, filemanager):
        self.llm = llm
        self.filemanager = filemanager
        self.align_targets = self.filemanager.load_settings()
        logging.info('Setting up samples files...')
        self.filemanager.setup_samples_files()
        logging.info('Samples files are set up.')
    
    def clustering_result_extraction(self):
        logging.info('Loading integrated labels...')
        integrated_labels = self.filemanager.load_integrated_labals()
        keys_labels = self.filemanager.load_keys_labels()
        keys_texts = self.filemanager.load_keys_texts()

        self.integration = {}
        for integrated_label in integrated_labels:
            label = integrated_label['Label']

            if label not in self.align_targets.keys():
                continue

            description = integrated_label['Description']
            self.integration[label] = {}
            self.integration[label]['Description'] = description
            self.integration[label]['Original_keys'] = []
            keys_indices = np.where(np.array(keys_labels) == label)[0]
            pmc_numbers = [keys_texts[n]['PMC_ID'] for n in keys_indices]
            for pmc_id in pmc_numbers:
                pmc_info = {'PMC_ID':pmc_id}
                pmc_info['Keys'] = [keys_texts[n]['Key'] for n in keys_indices if keys_texts[n]['PMC_ID'] == pmc_id]
                pmc_info['Keys_Info'] = [keys_texts[n]['Description'] for n in keys_indices if keys_texts[n]['PMC_ID'] == pmc_id]
                self.integration[label]['Original_keys'].append(pmc_info)
        
        self.filemanager.write_integration(self.integration)
        logging.info('Loading integrated labels...Done')
    
    def keyname_variations(self):
        logging.info('Extracting key names variations...')
        integrated_labels = self.filemanager.load_integrated_labals()
        keys_labels = self.filemanager.load_keys_labels()
        keys_texts = self.filemanager.load_keys_texts()

        key_name_variations = {}
        for integrated_label in integrated_labels:
            integrated_label_description = integrated_label['Description']
            integrated_label = integrated_label['Label']

            keys_indices = np.where(np.array(keys_labels) == integrated_label)[0]
            key_names = []
            for key_index in keys_indices:
                key_names.append(keys_texts[key_index]['Key'])
            key_name_variations[integrated_label] = {}
            key_name_variations[integrated_label]['Key_names'] = list(set(key_names))
            key_name_variations[integrated_label]['Description'] = integrated_label_description

        self.filemanager.write_keyname_variations(key_name_variations)
        logging.info('Extracting key names variations...Done.')
    
    def align_keys(self):
        logging.info('Aligning keys...')
        for target in self.align_targets.keys():
            logging.info(f'Aligning target: {target}')
            for original_key in self.integration[target]['Original_keys']:
                logging.info('\t'+str(original_key))

                pmc_number = original_key['PMC_ID']
                keys = original_key['Keys']
                keys_info = original_key['Keys_Info']

                samples = self.filemanager.load_samples_json(pmc_number)
                # Randomly sample 10 elements.
                random_indices = np.random.choice(len(samples), min(10, len(samples)), replace=False)
                samples_key_values = []
                for i in random_indices:
                    samples_key_values.append({key:samples[i].get(key) for key in keys})

                input_conditions = {}
                input_conditions['reference_keys'] = keys
                input_conditions['target_key'] = target
                input_conditions['reference_key_descriptions'] = keys_info
                input_conditions['target_key_description'] = self.align_targets[target]['Instructions']
                input_conditions['sample_values'] = samples_key_values

                result = self.llm.generate_transformation_code(input_conditions=input_conditions)
                result = json.loads(result)
                self.filemanager.write_transform_code(pmc_number, result)

                transform_code = result['Python_code']

                if 'Conversion_possible' in result.keys() and result['Conversion_possible'] == 'yes':
                    for i in range(len(samples)):

                        if not all([key in samples[i].keys() for key in keys]):
                            samples[i][f'EMBERS___{target}'] = None
                            continue

                        current_input = {key:samples[i][key] for key in keys}
                        local_vars = {'input': current_input}

                        try:
                            exec(transform_code, {}, local_vars)
                            transformed_value = local_vars['transform_data'](current_input)
                            samples[i][f'EMBERS___{target}'] = transformed_value
                        except Exception as e:
                            samples[i][f'EMBERS___{target}'] = None
                    
                self.filemanager.update_samples_json(pmc_number, samples)
            logging.info(f'Aligning target: {target}...Done')


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

    llm = LLM(api_key=Config.OPENAI_API_KEY, model_name=Config.MODEL_NAME)
    filemanager = FileManager(config=Config)
    aligner = Aligner(llm=llm, filemanager=filemanager)

    aligner.clustering_result_extraction()
    aligner.keyname_variations()
    aligner.align_keys()

    logging.info('End analyzing process.')