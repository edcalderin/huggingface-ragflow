import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline


class HuggingFaceModel:
    def _load_model(self) -> AutoModelForCausalLM:
        """
        The function `_load_model` loads a model for causal language modeling using
        Hugging Face's `AutoModelForCausalLM` with specific configuration settings.

        Returns:
        The `_load_model` function returns an instance of `AutoModelForCausalLM`
        loaded from the Hugging Face model with the specified configuration settings.
        """
        try:
            bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return AutoModelForCausalLM.from_pretrained(
                self._llm_config.model_name,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
        except Exception as ex:
            raise Exception(f"Error loading model from HF: {ex}") from ex

    def _create_pipeline(self, model: AutoModelForCausalLM) -> HuggingFacePipeline:
        """
        The function `_create_pipeline` creates a Hugging Face pipeline for text
        generation using a given model.

        Args:
        model: The `model` parameter in the `_create_pipeline` method is expected to
        be an instance of `AutoModelForCausalLM` class from the Hugging Face
        Transformers library. This model is used for text generation tasks in the
        pipeline.

        Returns:
        An instance of the `HuggingFacePipeline` class with the text generation
        pipeline created using the provided model and configuration parameters.
        """
        try:
            text_generation_pipeline: pipeline = pipeline(
                model=model,
                task=self._llm_config.task,
                temperature=self._llm_config.temperature,
                max_new_tokens=self._llm_config.max_new_tokens,
            )
            return HuggingFacePipeline(pipeline=text_generation_pipeline)
        except Exception as ex:
            raise (f"Error creating HF Pipeline: {ex}") from ex
