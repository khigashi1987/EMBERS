import os

class DevelopmentConfig:
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    COMPLETION_MODEL_NAME = 'gpt-4-turbo'
    MAX_TOKENS = 100000

    # EMBEDDING setting
    EMB_DIM_PROJECT = 3072
    EMB_DIM_METHODS = 3072
    EMB_DIM_KEYS = 512

    N_NEIGHBORS_PROJECT = 5
    MIN_DIST_PROJECT = 0.01

    N_NEIGHBORS_METHODS = 5
    MIN_DIST_METHODS = 0.01

    N_NEIGHBORS_KEYS = 50
    MIN_DIST_KEYS = 0.1

    # CLUSTERING setting
    RESOLUTION_PROJECT = 0.01
    RESOLUTION_METHODS = 0.01

    MIN_CLUSTER_SIZE_PROJECT = 10
    MIN_CLUSTER_SIZE_METHODS = 20

    KEYS_PURITY_THRESHOLD = 0.9
    KEYS_MIN_SIZE = 10

    DATA_DIR = '/Volumes/MDatahubDev/Total_result'

    OUT_DIR = '/Volumes/MDatahubDev/Total_result_integration/integrated'
    LOG_DIR = '/Volumes/MDatahubDev/Total_result_integration/log'

Config = DevelopmentConfig
