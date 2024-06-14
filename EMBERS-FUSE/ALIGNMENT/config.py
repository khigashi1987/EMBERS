import os

class DevelopmentConfig:
    INITIALIZE_FROM_ZERO = False
    
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MODEL_NAME = 'gpt-4-turbo'
    MAX_TOKENS = 100000

    DATA_DIR = '/Volumes/MDatahubDev/Total_result'
    INTEGRATED_DATA_DIR = '/Volumes/MDatahubDev/Total_result_integration/integrated'
    LOG_DIR = '/Volumes/MDatahubDev/Total_result/log'
    SETTING_FILE = './settings.json'

Config = DevelopmentConfig
