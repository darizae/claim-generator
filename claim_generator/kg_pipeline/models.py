import os
from openai import OpenAI

client = OpenAI()
from abc import ABC, abstractmethod
from typing import List

from ..kg_pipeline.config import ModelConfig, KGModelType
from ..kg_pipeline.prompt_templates import REFINED_CLAIM_PROMPT


class BaseKGModel(ABC):
    """
    Abstract base class for KG-model backends.
    Must return a JSON with a 'triples' array.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate_kg(self, texts: List[str]) -> List[str]:
        """Given plain texts, build the prompt internally and return JSON strings."""
        pass

    @abstractmethod
    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        """
        Similar but user provides the full prompts.
        Returns raw JSON strings with a 'triples' array.
        """
        pass

    @staticmethod
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


class OpenAIKGModel(BaseKGModel):
    """
    Minimal OpenAI-based knowledge graph parser.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        openai_api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not set.")
        # We'll call openai directly using openai.ChatCompletion calls.

    def generate_kg(self, texts: List[str]) -> List[str]:
        """
        If user passes raw texts, we build the default REFINED_CLAIM_PROMPT for each text.
        """
        results = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = REFINED_CLAIM_PROMPT.format(input=text)
                content = self._call_openai_chat(prompt)
                results.append(content)
        return results

    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        """
        If user passes already-customized prompts, we just call the model with them.
        """
        results = []
        for batch in self.chunked(prompts, self.config.batch_size):
            for prompt in batch:
                content = self._call_openai_chat(prompt)
                results.append(content)
        return results

    def _call_openai_chat(self, prompt: str) -> str:
        response = client.chat.completions.create(model=self.config.model_name_or_path,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.config.temperature)
        return response.choices[0].message.content
