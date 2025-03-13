from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    filename: str = field(default="AI Engineer-Lead - Anexo.pdf")
    embedding_model_name: str = field(default="sentence-transformers/all-mpnet-base-v2")
    collection_name: str = field(default="historiacard_docs")
    qdrant_store_path: str = field(default="./tmp")

    # Retrieval
    model_name: str = field(default="meta-llama/Llama-2-7b-chat-hf")
    task: str = field(default="text-generation")
    temperature: float = field(default=0.2)
