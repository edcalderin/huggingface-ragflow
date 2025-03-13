from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    filename: str = Field(default="AI Engineer-Lead - Anexo.pdf", frozen=True)
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", frozen=True
    )
    collection_name: str = Field(default="historiacard_docs", frozen=True)
    qdrant_store_path: str = Field(
        default="./tmp", alias="Directory to store embeddings", frozen=True
    )
