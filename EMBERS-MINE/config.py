import os
import json

class DevelopmentConfig:
    # LLM setting
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MODEL_NAME = 'gpt-4-turbo'
    MAX_TOKENS = 120000

    PMC_DIR = './PMC_Dataset'
    RESULT_BASE_DIR = './result'
    LOG_DIR = './log'

    USER_EMAIL = os.environ.get('ENTREZ_EMAIL')

    SKIP_SUPP_SIZE = 20 * 1024 * 1024  # 20MB

    SCHEMA_PROJECT_JSON = './SCHEMA/SCHEMA_project_update.json'

    DATABASE_RELATED_KEYS_JSON = './SCHEMA/DB_related_terms.json'
    DATABASE_RELATED_KEYS = json.load(open(DATABASE_RELATED_KEYS_JSON))

Config = DevelopmentConfig
