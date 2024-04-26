import os

class DevelopmentConfig:
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MODEL_NAME = 'gpt-4-turbo'
    MAX_TOKENS = 120000

    # EMBEDDING setting
    EMB_DIM_PROJECT = 3072
    EMB_DIM_METHODS = 3072
    EMB_DIM_KEYS = 3072
    N_NEIGHBORS = 5
    MIN_DIST = 0.01

    # CLUSTERING setting
    PROJECT_MIN_CLUSTER_SIZE = 10

    DATA_DIR = '/Volumes/MDatahubDev/Total_result'

    OUT_DIR = '/Volumes/MDatahubDev/Total_result_integration/integrated'
    LOG_DIR = '/Volumes/MDatahubDev/Total_result_integration/log'

Config = DevelopmentConfig
