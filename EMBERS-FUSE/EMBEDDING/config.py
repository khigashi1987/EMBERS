import os
import json

class DevelopmentConfig:
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MODEL_NAME = 'text-embedding-3-large'

    RESULT_BASE_DIR = '/Volumes/MDatahubDev/Total_result'
    LOG_DIR = '/Volumes/MDatahubDev/Total_result/log'

Config = DevelopmentConfig
