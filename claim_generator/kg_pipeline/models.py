import os
import requests
import openai
from abc import ABC, abstractmethod
from typing import List

# For HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

from .config import ModelConfig, KGModelType
from .prompt_templates import REFINED_CLAIM_PROMPT


class BaseKGModel(ABC):
    """
    Abstract base class for KG-model backends.
    Must return a JSON with a 'triples' array for each input.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate_kg(self, texts: List[str]) -> List[str]:
        """Given plain texts, build the prompt internally and return JSON strings."""
        raise NotImplementedError

    @abstractmethod
    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        """
        Similar but user provides the full prompts.
        Returns raw JSON strings with a 'triples' array.
        """
        raise NotImplementedError

    @staticmethod
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


class HuggingFaceKGModel(BaseKGModel):
    """
    HuggingFace-based model that tries causal or seq2seq models.
    Outputs are expected to contain well-formed JSON with a 'triples' array.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        # Attempt to load as a causal model, fallback to seq2seq:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)

        self.model.to(self.config.device)

        # Some HF models need a pad token set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_kg(self, texts: List[str]) -> List[str]:
        """
        Build default REFINED_CLAIM_PROMPT for each text and pass to HF model.
        """
        outputs = []
        for batch in self.chunked(texts, self.config.batch_size):
            prompts = [REFINED_CLAIM_PROMPT.format(input=t) for t in batch]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            # Generate
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0),
            )

            decoded_batch = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)
            outputs.extend(decoded_batch)

        return outputs

    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        """
        If user has already built custom prompts, pass them directly.
        """
        outputs = []
        for batch in self.chunked(prompts, self.config.batch_size):
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0),
            )

            decoded_batch = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)
            outputs.extend(decoded_batch)

        return outputs


class OpenAIKGModel(BaseKGModel):
    """
    OpenAI-based knowledge graph parser using ChatCompletion.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Attempt to get API key from config or environment:
        openai_api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in config or environment variables.")
        # Set it on the openai module:
        openai.api_key = openai_api_key

    def generate_kg(self, texts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = REFINED_CLAIM_PROMPT.format(input=text)
                response = openai.ChatCompletion.create(
                    model=self.config.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature
                )
                content = response.choices[0].message.content
                results.append(content)
        return results

    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(prompts, self.config.batch_size):
            for prompt in batch:
                response = openai.ChatCompletion.create(
                    model=self.config.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature
                )
                content = response.choices[0].message.content
                results.append(content)
        return results


class JanLocalKGModel(BaseKGModel):
    """
    Example local LLM server usage.
    Only needed if you want to keep the 'jan_local' approach from original code.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.endpoint_url:
            raise ValueError("endpoint_url must be provided for JanLocalKGModel.")

    def generate_kg(self, texts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = REFINED_CLAIM_PROMPT.format(input=text)
                content = self._call_local_api(prompt)
                results.append(content)
        return results

    def generate_kg_prompts(self, prompts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(prompts, self.config.batch_size):
            for prompt in batch:
                content = self._call_local_api(prompt)
                results.append(content)
        return results

    def _call_local_api(self, prompt: str) -> str:
        payload = {
            "model": self.config.model_name_or_path,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.config.endpoint_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]
