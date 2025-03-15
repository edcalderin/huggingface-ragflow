import logging
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logging import setup_logging

load_dotenv()

login(token=os.getenv("HUGGINGFACE_TOKEN"))
setup_logging()


class Embedding:
    @staticmethod
    def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        """
        The function `_load_embeddings` returns a `HuggingFaceEmbeddings` object with
        a specified model name and location.

        Returns:
        An instance of the `HuggingFaceEmbeddings` class.
        """
        try:
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logging.info(f"Device: {device}")

            return HuggingFaceEmbeddings(
                model_name=model_name,
                multi_process=True,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as ex:
            raise Exception(f"Error loading HuggingFace embeddings: {ex}") from ex
