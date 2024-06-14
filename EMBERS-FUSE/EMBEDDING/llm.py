import numpy as np
import openai
from config import Config

class LLM():
    ###
    # LLM関連の処理を実行するクラス
    ###
    def __init__(self, 
                 api_key='', 
                 model_name='text-embedding-3-large'):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        
        response = self.client.embeddings.create(input = [text],
                                                 model=self.model_name)
        try:
            embedding = response.data[0].embedding
            embedding = np.array(embedding)
        except Exception as e:
            print(e)
            embedding = None
        return embedding
    
    def get_multiple_embedding(self, text_list):
        text_list = [text.replace("\n", " ") for text in text_list]
        response = self.client.embeddings.create(input = text_list,
                                                 model=self.model_name)
        try:
            embeddings = [d.embedding for d in response.data]
            embeddings = np.array(embeddings)
        except Exception as e:
            print(e)
            embeddings = None
        return embeddings

if __name__ == '__main__':
    llm = LLM(api_key=Config.OPENAI_API_KEY, model_name=Config.MODEL_NAME)
    test_messages = ["This is a test message.", "National Institute of Genetics, Japan"]
    embeddings = llm.get_multiple_embedding(test_messages)
    print(type(embeddings))
    print(embeddings.shape)
