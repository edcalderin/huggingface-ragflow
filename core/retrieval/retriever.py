import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from core.config import LLMConfig
from core.model import Embedding
from utils.logging import setup_logging

setup_logging()


class Retriever:
    @staticmethod
    def retrieve():
        try:
            embeddings: HuggingFaceEmbeddings = Embedding.load_embeddings(
                model_name=LLMConfig.EMBEDDING_MODEL_NAME
            )
            logging.info("Loading Qdrant retriever")
            client = QdrantClient(path=LLMConfig.QDRANT_STORE_PATH)
            vector_store = QdrantVectorStore(
                client=client,
                embedding=embeddings,
                collection_name=LLMConfig.COLLECTION_NAME,
            )
            return vector_store.as_retriever()
        except Exception as ex:
            logging.error("Error by getting retriever")
            raise ex
