from typing import List
from zhipuai import ZhipuAI
from langchain.embeddings.base import Embeddings

API_KEY = 'fc4c27cb51ee4722a642b76909d01464.czerMWREZH1xkavS'

def zhipu_embedding(text):
    client = ZhipuAI(api_key=API_KEY)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    embedding_vect = [d for d in response.data]
    return embedding_vect

class ZhipuAIEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return zhipu_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        return zhipu_embedding(text)