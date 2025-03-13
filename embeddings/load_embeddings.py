import asyncio
import logging
from pathlib import Path

import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from embeddings.config import EmbeddingConfig

logging.basicConfig(
    level=logging.INFO, format="%(name)s :: %(levelname)s :: %(message)s"
)

path_dir: Path = Path(__file__).parent / "Documents"


class EmbeddingLoader:
    def __init__(self) -> None:
        self._embedding_config = EmbeddingConfig()

    def _get_documents(self) -> list[Document]:
        """Get Documents.

        Extract document from pdf file.

        Returns:
        List of Documents
        """
        filename: Path = path_dir / self._embedding_config.filename
        loader = PyPDFLoader(filename)
        docs = [doc for doc in loader.lazy_load()]
        return docs

    @staticmethod
    def _create_chunks(docs: list[Document]) -> list[Document]:
        """
        The function `_create_chunks` takes a list of `Document` objects, splits the
        text content of each document using a `CharacterTextSplitter`, and returns a
        list of split `Document` objects.

        Args:
          docs: The `docs` parameter is a list of `Document` objects that are being
        passed to the `_create_chunks` method.

        Returns:
          The function `_create_chunks` is returning a list of documents that have
          been split using the `CharacterTextSplitter` class.
        """
        text_splitter = CharacterTextSplitter()
        return text_splitter.split_documents(docs)

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """
        The function `_load_embeddings` returns a `HuggingFaceEmbeddings` object with
        a specified model name and location.

        Returns:
          An instance of the `HuggingFaceEmbeddings` class with the specified model
        name and location set to ":memory:".
        """
        try:
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logging.info(f"Device: {device}")

            return HuggingFaceEmbeddings(
                model_name=self._embedding_config.embedding_model_name,
                multi_process=True,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as ex:
            raise Exception(f"Error loading HuggingFace embeddings: {ex}") from ex

    async def _async_store_documents(
        self, documents: list[Document], embeddings: HuggingFaceEmbeddings
    ) -> None:
        """
        This async function stores documents with embeddings in a Qdrant vector store.

        Args:
          documents: The `documents` parameter is a list of `Document` objects that
        contain the data to be stored in the vector store.
          embeddings: The `embeddings` parameter in the `_async_store_documents`
        function is of type `HuggingFaceEmbeddings`. It is used to provide the
        embeddings for the documents that are being stored in the vector store.
        """
        try:
            await QdrantVectorStore.afrom_documents(
                documents=documents,
                path=self._embedding_config.qdrant_store_path,
                collection_name=self._embedding_config.collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
            )
        except Exception as ex:
            raise Exception(f"Error by initializing vector embeddings: {ex}") from ex

    async def load_to_qdrant_index(self) -> None:
        try:
            logging.info("Starting...")
            documents: list[Document] = self._get_documents()
            chunked_documents: list[Document] = self._create_chunks(documents)
            embeddings: HuggingFaceEmbeddings = self._load_embeddings()
            await self._async_store_documents(chunked_documents, embeddings)
        except Exception as e:
            logging.error(e)
            raise e
        else:
            logging.info("Embeddings stored successfully!")


if __name__ == "__main__":
    embedding_loader = EmbeddingLoader()
    asyncio.run(embedding_loader.load_to_qdrant_index())
