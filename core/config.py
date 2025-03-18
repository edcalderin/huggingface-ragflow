from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    FILENAME: str = "AI Engineer-Lead - Anexo.pdf"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    COLLECTION_NAME: str = "historiacard_docs"
    QDRANT_STORE_PATH: str = "./tmp"

    # Model
    MODEL_NAME: str = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_TASK: str = "text-generation"
    TEMPERATURE: float = 0.1
    MAX_NEW_TOKENS: int = 1024
