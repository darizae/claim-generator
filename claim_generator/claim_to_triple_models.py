# claim_generator/claim_to_triple_models.py

import json
import requests
from abc import ABC, abstractmethod
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .config import ModelConfig, ModelType
from .utils import retry_with_exponential_backoff

# This can be a system prompt or inline instruction for converting a single claim => triple dict
CLAIM_TO_TRIPLE_SYSTEM_PROMPT = """You are an expert at mapping a short factual statement into a knowledge graph triple.
Output valid JSON with exactly three keys: "subject", "predicate", "object". No extra text.
"""


def build_claim_to_triple_prompt(claim: str) -> str:
    """
    Wrap the claim in a user prompt instructing the model to produce a triple in JSON.
    """
    user_prompt = f"""Claim: "{claim}"
Return only JSON, for example:
{{
  "subject": "some subject",
  "predicate": "some relation",
  "object": "some object"
}}"""
    # If you want a single string, you can embed system vs user prompts together,
    # or handle them as “messages” for a chat model.  This example returns just user text:
    return user_prompt


###############################################################################
# Base class
###############################################################################
class BaseClaimToTripleModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def convert_claims(self, claims: List[str]) -> List[Dict[str, str]]:
        """
        Given a list of claim strings, return a list of triple dicts:
            [
              {"subject": "...", "predicate": "...", "object": "..."},
              ...
            ]
        The length of the output list should match the input claims list (1:1).
        """
        pass


###############################################################################
# HuggingFace-based
###############################################################################
class HuggingFaceClaimToTripleModel(BaseClaimToTripleModel):
    """
    Uses a local or HF-hosted model, which might be causal or seq2seq.
    We'll do one forward pass per claim for simplicity, or do a batch if you prefer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # Attempt causal first, fallback to seq2seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)

        self.model.to(self.config.device)
        # Set pad token if needed
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def convert_claims(self, claims: List[str]) -> List[Dict[str, str]]:
        triples = []
        for claim in claims:
            prompt = build_claim_to_triple_prompt(claim)

            inputs = self.tokenizer(
                CLAIM_TO_TRIPLE_SYSTEM_PROMPT + "\n" + prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0),
            )
            decoded = self.tokenizer.decode(gen_output[0], skip_special_tokens=True).strip()

            # Attempt to parse as JSON
            triple_dict = self._parse_json_triple(decoded)
            triples.append(triple_dict)
        return triples

    def _parse_json_triple(self, text: str) -> Dict[str, str]:
        """
        Attempt to parse the model output as a single JSON object with
        keys "subject", "predicate", "object".
        """
        try:
            data = json.loads(text)
            # Minimal validation
            if all(k in data for k in ("subject", "predicate", "object")):
                return {
                    "subject": data["subject"],
                    "predicate": data["predicate"],
                    "object": data["object"]
                }
        except json.JSONDecodeError:
            pass
        # Fallback
        return {"subject": "", "predicate": "", "object": ""}


###############################################################################
# JanLocal-based
###############################################################################
class JanLocalClaimToTripleModel(BaseClaimToTripleModel):
    """
    Hits a local “Jan” server, which mimics the OpenAI ChatCompletion API style.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.endpoint_url:
            raise ValueError("endpoint_url must be specified for JanLocalClaimToTripleModel")

    @retry_with_exponential_backoff
    def _call_jan_server(self, payload, headers):
        resp = requests.post(
            self.config.endpoint_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def convert_claims(self, claims: List[str]) -> List[Dict[str, str]]:
        triples = []
        for claim in claims:
            user_prompt = build_claim_to_triple_prompt(claim)
            messages = [
                {"role": "system", "content": CLAIM_TO_TRIPLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            payload = {
                "model": self.config.model_name_or_path,
                "messages": messages,
                "temperature": self.config.temperature
            }
            headers = {"Content-Type": "application/json"}

            data = self._call_jan_server(payload, headers)
            content = data["choices"][0]["message"]["content"].strip() if data.get("choices") else ""

            triple_dict = self._parse_json_triple(content)
            triples.append(triple_dict)
        return triples

    def _parse_json_triple(self, text: str) -> Dict[str, str]:
        # same logic as in HF
        try:
            data = json.loads(text)
            if all(k in data for k in ("subject", "predicate", "object")):
                return {
                    "subject": data["subject"],
                    "predicate": data["predicate"],
                    "object": data["object"]
                }
        except json.JSONDecodeError:
            pass
        # fallback
        return {"subject": "", "predicate": "", "object": ""}


def create_claim_to_triple_model(config: ModelConfig) -> BaseClaimToTripleModel:
    if config.model_type == ModelType.HUGGINGFACE:
        return HuggingFaceClaimToTripleModel(config)
    elif config.model_type == ModelType.JAN_LOCAL:
        return JanLocalClaimToTripleModel(config)
    else:
        raise ValueError(f"Unsupported model type for claim-to-triple retrofit: {config.model_type}")
