import os
import json

class DevelopmentConfig:
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MODEL_NAME = 'text-embedding-3-large'
    # OpenAI latest embedding model (text-embedding-3-large/small) show poor performance of keys embeddings for some reason.
    # When I tried to use the previous model (text-embedding-ada-002), it worked well.
    MODEL_NAME_FOR_KEYS = 'text-embedding-ada-002'

    RESULT_BASE_DIR = '/Volumes/MDatahubDev/Total_result'
    LOG_DIR = '/Volumes/MDatahubDev/Total_result/log'

Config = DevelopmentConfig
