from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    model_config = {
        "frozen": True,  # Makes the entire model immutable
    }
    filename: str = Field(
        default="AI Engineer-Lead - Anexo.pdf", description="Document used for this RAG"
    )
    embedding_model_name: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    collection_name: str = Field(default="historiacard_docs")
    qdrant_store_path: str = Field(
        default="./tmp", description="Directory to store embeddings"
    )

    # Retrieval
    model_name: str = Field(default="meta-llama/Llama-2-7b-chat-hf")
    task: str = Field(default="text-generation")
    temperature: float = Field(default=0.2)
