import os
import shutil
import glob
import pickle
import json
import numpy as np
import logging

class FileManager():
    def __init__(self, config):
        self.data_dir = config.DATA_DIR
        self.integration_dir = config.INTEGRATED_DATA_DIR
        self.log_dir = config.LOG_DIR
        self.instructions_file = config.INSTRUCTIONS_FILE
        self.initialize_from_zero = config.INITIALIZE_FROM_ZERO
    
    def setup_samples_files(self):
        if self.initialize_from_zero:
            for samples_file in glob.glob(os.path.join(self.data_dir, 'PMC*/PMC*_samples_update.json')):
                pmc_number = samples_file.split('/')[-1].split('_')[0]
                output_file = os.path.join(self.data_dir, f'{pmc_number}/{pmc_number}_samples_update_integrated.json')
                shutil.copy(samples_file, output_file)
        else:
            pass

    def load_instructions(self):
        with open(self.instructions_file, 'r') as f:
            instructions = json.load(f)
        return instructions

    def load_integrated_labals(self):
        keys_integrated_labels_file = os.path.join(self.integration_dir, 'keys_labels_descriptions.json')
        keys_integrated_labels = json.load(open(keys_integrated_labels_file, 'r'))
        return keys_integrated_labels
    
    def load_keys_labels(self):
        keys_labels_file = os.path.join(self.integration_dir, 'keys_labels.pkl')
        with open(keys_labels_file, 'rb') as f:
            keys_labels = pickle.load(f)
        return keys_labels
    
    def load_keys_texts(self):
        keys_texts_file = os.path.join(self.integration_dir, 'keys_texts.pkl')
        with open(keys_texts_file, 'rb') as f:
            keys_texts = pickle.load(f)
        return keys_texts
    
    def check_if_samples_json_exists(self, pmc_number):
        samples_json_file = os.path.join(self.data_dir, f'{pmc_number}/{pmc_number}_samples_update_integrated.json')
        return os.path.exists(samples_json_file)
    
    def load_samples_json(self, pmc_number):
        samples_json_file = os.path.join(self.data_dir, f'{pmc_number}/{pmc_number}_samples_update_integrated.json')
        with open(samples_json_file, 'r') as f:
            samples_json = json.load(f)
        return samples_json
    
    def update_samples_json(self, pmc_number, samples_json):
        samples_json_file = os.path.join(self.data_dir, f'{pmc_number}/{pmc_number}_samples_update_integrated.json')
        with open(samples_json_file, 'w') as f:
            json.dump(samples_json, f, indent=4)
    
    def write_integration(self, integration):
        output_file = os.path.join(self.integration_dir, 'integration.json')
        with open(output_file, 'w') as f:
            json.dump(integration, f, indent=4)

    def write_keyname_variations(self, keyname_variations):
        output_file = os.path.join(self.integration_dir, 'key_name_variations.json')
        with open(output_file, 'w') as f:
            json.dump(keyname_variations, f, indent=4)

    def write_transform_code(self, key, pmc_number, transform_code_result):
        output_file = os.path.join(self.data_dir, f'{pmc_number}/{pmc_number}_transform_code.json')
        if not os.path.exists(output_file):
            out_content = {}
            out_content[key] = transform_code_result
            with open(output_file, 'w') as f:
                json.dump(out_content, f, indent=4)
        else:
            previous_content = json.load(open(output_file, 'r'))
            previous_content[key] = transform_code_result
            with open(output_file, 'w') as f:
                json.dump(previous_content, f, indent=4)
