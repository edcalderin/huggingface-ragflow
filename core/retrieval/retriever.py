import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from core.config import LLMConfig
from core.model.embedding import Embedding
from utils.logging import setup_logging

setup_logging()


class Retriever:
    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config: LLMConfig = llm_config
        self._embeddings: HuggingFaceEmbeddings = Embedding.load_embeddings(
            model_name=self._llm_config.embedding_model_name
        )

    def retrieve(self):
        try:
            logging.info("Loading Qdrant retriever")
            client = QdrantClient(path=self._llm_config.qdrant_store_path)
            vector_store = QdrantVectorStore(
                client=client,
                embedding=self._embeddings,
                collection_name=self._llm_config.collection_name,
            )
            return vector_store.as_retriever()
        except Exception as ex:
            logging.error("Error by getting retriever")
            raise ex


if __name__ == "__main__":
    Retriever(llm_config=LLMConfig()).retrieve()
