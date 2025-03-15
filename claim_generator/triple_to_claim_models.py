import os

import openai
import requests
from openai import OpenAI

from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from .config import ModelConfig
from .prompts import TRIPLE_TO_CLAIM_PROMPT
from .utils import retry_with_exponential_backoff


###############################################################################
# Base class
###############################################################################
class BaseTripleToClaimModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def triple_to_claim(self, subject: str, predicate: str, obj: str) -> str:
        """Convert one triple into a single short text claim."""
        pass


###############################################################################
# HuggingFace-based model
###############################################################################
class HuggingFaceTripleToClaimModel(BaseTripleToClaimModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Try CausalLM first; if that fails, fallback to Seq2Seq
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)

        self.model.to(self.config.device)
        # Set pad token if needed
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def triple_to_claim(self, subject: str, predicate: str, obj: str) -> str:
        prompt = TRIPLE_TO_CLAIM_PROMPT.format(subject=subject, predicate=predicate, object=obj)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)

        gen_output = self.model.generate(
            **encoded,
            max_new_tokens=64,
            temperature=self.config.temperature,
            do_sample=(self.config.temperature > 0),
        )
        decoded = self.tokenizer.decode(gen_output[0], skip_special_tokens=True)
        # Basic cleaning: for many models, the entire system prompt might get echoed, so trim
        return self._postprocess(decoded)

    def _postprocess(self, text: str) -> str:
        # Minimal cleanup
        return text.strip().split(")")[-1].strip()  # naive; adjust as needed


###############################################################################
# OpenAI-based model
###############################################################################
class OpenAITripleToClaimModel(BaseTripleToClaimModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        openai_api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is missing in config or environment.")

    @retry_with_exponential_backoff
    def _chat_completion(self, messages):
        client = OpenAI(api_key=self.config.api_key)
        return client.chat.completions.create(model=self.config.model_name_or_path,
                                              messages=messages,
                                              temperature=self.config.temperature)

    def triple_to_claim(self, subject: str, predicate: str, obj: str) -> str:
        prompt = TRIPLE_TO_CLAIM_PROMPT.format(subject=subject, predicate=predicate, object=obj)
        messages = [
            {"role": "system", "content": "Convert triple to short factual statement."},
            {"role": "user", "content": prompt}
        ]
        response = self._chat_completion(messages)
        return response.choices[0].message.content.strip()


###############################################################################
# Jan-local-based model
###############################################################################
class JanLocalTripleToClaimModel(BaseTripleToClaimModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not self.config.endpoint_url:
            raise ValueError("endpoint_url must be set for JanLocalTripleToClaimModel.")

    @retry_with_exponential_backoff
    def _call_jan_server(self, payload, headers):
        return requests.post(self.config.endpoint_url, json=payload, headers=headers, timeout=30)

    def triple_to_claim(self, subject: str, predicate: str, obj: str) -> str:
        prompt = TRIPLE_TO_CLAIM_PROMPT.format(subject=subject, predicate=predicate, object=obj)
        payload = {
            "model": self.config.model_name_or_path,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
        }
        headers = {"Content-Type": "application/json"}
        resp = self._call_jan_server(payload, headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return f"{subject} {predicate} {obj}"
        return choices[0]["message"]["content"].strip()
