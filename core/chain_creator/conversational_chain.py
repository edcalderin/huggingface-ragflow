import logging
from operator import itemgetter
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from core.model import LLMModel
from core.retrieval import Retriever
from utils.logging import setup_logging

setup_logging()


class ConversationalChain:
    def __init__(self) -> None:
        self._llm_model: LLMModel = LLMModel()

    def _get_prompt_template(self) -> ChatPromptTemplate:
        try:
            prompt_path: Path = Path(__file__).parent / "prompt.txt"
            prompt: str = prompt_path.read_text()
            return ChatPromptTemplate.from_template(prompt)
        except FileNotFoundError as ex:
            raise ex

    def get_chain(self) -> RunnableSequence:
        retriever = Retriever.retrieve()
        llm = self._llm_model.create_pipeline()
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | self._get_prompt_template()
            | llm
            | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    response = (
        ConversationalChain()
        .get_chain()
        .invoke({"question": "Cual es la mision de HistoriaCard?"})
    )
    logging.info(response)
