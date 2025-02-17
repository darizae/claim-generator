import json
import os
import requests
from abc import ABC, abstractmethod
from typing import List, Tuple

import openai
from openai import OpenAI

client = OpenAI()
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from .config import ModelType
from .prompts import PromptTemplate, get_prompt_template
from .triple_to_claim_models import *

from kg_parser import KGParser


class BaseClaimGenerator(ABC):
    """
    Abstract base class for all claim generators.
    """

    def __init__(self, config: ModelConfig, prompt_template: PromptTemplate, granularity: str = None):
        self.config = config
        self.prompt_template = prompt_template
        self.granularity = granularity

    @abstractmethod
    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        """
        Return a list of lists of claims, matching the order of input `texts`.
        """
        pass

    def build_claim_extraction_prompt(self, text: str, granularity: str = None) -> str:
        """
        Inject the `text` into the selected prompt template. Optionally pass a `granularity`
        argument if the prompt requires it (e.g., PromptTemplate.GRANULARITY).
        """
        # If you only want to pass granularity when the template is GRANULARITY-based:
        if self.prompt_template == PromptTemplate.GRANULARITY and granularity:
            template_str = get_prompt_template(
                self.prompt_template,
                SOURCE_TEXT=text,
                granularity=granularity
            )
        else:
            template_str = get_prompt_template(self.prompt_template, SOURCE_TEXT=text)

        return template_str

    @staticmethod
    def chunked(iterable, size: int):
        """
        Yield successive `size`-sized chunks from `iterable`.
        """
        for i in range(0, len(iterable), size):
            yield iterable[i: i + size]

    @staticmethod
    def parse_json_output(output_str: str) -> List[str]:
        """
        Attempt to parse JSON with a top-level "claims" field => list of claims.
        Return [] on error or no claims found.
        """
        try:
            data = json.loads(output_str.strip())
            claims = data.get("claims", [])
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            return []


class HuggingFaceSeq2SeqGenerator:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        self.model.to(config.device)

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i: i + self.config.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.config.temperature
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                # Example: parse JSON if your model outputs JSON, or do something else
                claims = self._parse_json_output(d)
                results.append(claims)
        return results

    def _parse_json_output(self, text: str) -> List[str]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "claims" in data:
                return data["claims"]
            return [data] if isinstance(data, str) else []
        except json.JSONDecodeError:
            # Fallback: perhaps split on newlines or periods
            return [line for line in text.split(".") if line.strip()]


class HuggingFaceCausalGenerator(BaseClaimGenerator):
    """
    Generator that uses a causal (decoder-only) HF model.
    """

    def __init__(self, config: ModelConfig, prompt_template: PromptTemplate, granularity: str = None):
        super().__init__(config, prompt_template, granularity)
        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        tokenizer_cls = AutoTokenizer
        model_cls = AutoModelForCausalLM

        tokenizer = tokenizer_cls.from_pretrained(self.config.model_name_or_path)
        model = model_cls.from_pretrained(self.config.model_name_or_path)

        # Set pad token if needed
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

        model.to(self.config.device)
        return tokenizer, model

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in self.chunked(texts, self.config.batch_size):
            prompts = [self.build_claim_extraction_prompt(t, self.granularity) for t in batch]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0.0),
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for d in decoded:
                claims = self.parse_json_output(d)
                all_claims.append(claims)

        return all_claims


class OpenAIClaimGenerator(BaseClaimGenerator):
    def __init__(self, config: ModelConfig, prompt_template: PromptTemplate, granularity: str = None):
        super().__init__(config, prompt_template, granularity)
        openai_api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key, timeout=30)
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found in config or environment.")

    @retry_with_exponential_backoff
    def _call_openai_chat(self, prompt):
        """
        Isolated function for the actual chat.completions.create call,
        so it can be wrapped in a retry decorator.
        """
        response = client.chat.completions.create(
            model=self.config.model_name_or_path,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,

        )
        return response

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = self.build_claim_extraction_prompt(text, self.granularity)
                response = self._call_openai_chat(prompt)
                content = response.choices[0].message.content
                claims = self.parse_json_output(content)
                all_claims.append(claims)
        return all_claims


class JanLocalClaimGenerator(BaseClaimGenerator):
    """
    Example for a local “Jan” server that mimics OpenAI's API spec.
    """

    def __init__(self, config: ModelConfig, prompt_template: PromptTemplate, granularity: str = None):
        super().__init__(config, prompt_template, granularity)
        if not config.endpoint_url:
            raise ValueError("endpoint_url must be provided for JanLocalClaimGenerator.")

    @retry_with_exponential_backoff
    def _call_jan_server(self, payload, headers):
        """
        Isolated function for the actual requests.post call,
        so it can be decorated with retry logic.
        """
        return requests.post(
            self.config.endpoint_url,
            json=payload,
            headers=headers,
            timeout=30
        )

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = self.build_claim_extraction_prompt(text, self.granularity)
                payload = {
                    "model": self.config.model_name_or_path,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                }
                headers = {"Content-Type": "application/json"}

                resp = self._call_jan_server(payload, headers)
                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    all_claims.append([])
                    continue

                content = choices[0]["message"]["content"]
                claims = self.parse_json_output(content)
                all_claims.append(claims)

        return all_claims


class KGToClaimsGenerator(BaseClaimGenerator):
    """
    A pipeline generator that:
      1) Uses KGParser to extract triples from raw text.
      2) Converts each triple to a natural language claim.

    It also exposes intermediate information for synthetic logging:
      - The original text ("reference")
      - The full prompt sent to kg-parser
      - The KG parser's output (in dict form)
      - The full triple-to-claim prompt for each triple
      - The array of claims produced
    """

    def __init__(self, config: ModelConfig):
        # We ignore the prompt_template since we use the internal kg-parser prompt.
        super().__init__(config, PromptTemplate.DEFAULT)
        from kg_parser import ModelType as KGModelType
        config.model_type = KGModelType.OPENAI
        self.kg_parser = KGParser(config)
        config.model_type = ModelType.OPENAI
        self.triple_to_claim_model = self._initialize_triple_to_claim_model(config)

    def _initialize_triple_to_claim_model(self, config: ModelConfig) -> BaseTripleToClaimModel:
        if config.model_type == ModelType.HUGGINGFACE:
            return HuggingFaceTripleToClaimModel(config)
        elif config.model_type == ModelType.OPENAI:
            return OpenAITripleToClaimModel(config)
        elif config.model_type == ModelType.JAN_LOCAL:
            return JanLocalTripleToClaimModel(config)
        else:
            raise ValueError("KGToClaimsGenerator expects HF, OPENAI, or JAN_LOCAL model_type for triple→claim step.")

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        """
        Regular method returning just the claims.
        """
        _, claims = self.generate_claims_with_intermediates(texts)
        return claims

    def generate_claims_with_intermediates(self, texts: List[str]) -> Tuple[List[dict], List[List[str]]]:
        """
        For each input text, this method returns a synthetic dictionary with:
          - 'reference': the original text
          - 'kg_parser_prompt': the full prompt used by kg-parser
          - 'kg_output': the KG parser's output (triples in dict form)
          - 'triple_to_claim_prompts': a list of the full prompts used for converting each triple
          - 'claims': a list of natural language claims (one per triple)

        Also returns a list of claim arrays (one array per input) for compatibility.
        """
        # Import the KG parser prompt template.
        from kg_parser.prompt_templates import REFINED_CLAIM_PROMPT

        kg_outputs = self.kg_parser.parse_batch(texts)
        synthetic_outputs = []
        all_claims = []
        for text, kg_out in zip(texts, kg_outputs):
            # Compute the full prompt used in kg-parser:
            kg_parser_prompt = REFINED_CLAIM_PROMPT.format(input=text)
            # Convert the KGOutput into dict form:
            kg_output_dict = {"triples": [vars(triple) for triple in kg_out.triples]}
            triple_prompts = []
            claims_for_text = []
            for triple in kg_out.triples:
                # Compute the triple-to-claim prompt using our TRIPLE_TO_CLAIM_PROMPT template:
                triple_prompt = TRIPLE_TO_CLAIM_PROMPT.format(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object
                )
                triple_prompts.append(triple_prompt)
                # Generate the claim for this triple:
                claim_text = self.triple_to_claim_model.triple_to_claim(
                    triple.subject,
                    triple.predicate,
                    triple.object
                )
                claims_for_text.append(claim_text)
            synthetic_entry = {
                "reference": text,
                "kg_parser_prompt": kg_parser_prompt,
                "kg_output": kg_output_dict,
                "triple_to_claim_prompts": triple_prompts,
                "claims": claims_for_text
            }
            synthetic_outputs.append(synthetic_entry)
            all_claims.append(claims_for_text)
        return synthetic_outputs, all_claims


def create_generator(
        config: ModelConfig,
        prompt_template: PromptTemplate,
        granularity: str = None
) -> BaseClaimGenerator:
    """
    Factory function that returns the appropriate claim generator
    based on config.model_type.
    """
    if config.model_type == ModelType.HUGGINGFACE:
        # Heuristic: If the model is "seq2seq" vs "causal."
        # A simple approach might be to see if "gpt" or "llama" is in the name => causal,
        # else default to seq2seq.
        # But you can do more robust checks or add a separate config field if you prefer.
        lower_name = config.model_name_or_path.lower()
        if any(x in lower_name for x in ["gpt", "llama", "opt", "falcon"]):
            return HuggingFaceCausalGenerator(config, prompt_template, granularity)
        else:
            return HuggingFaceSeq2SeqGenerator(config)
    elif config.model_type == ModelType.OPENAI:
        return OpenAIClaimGenerator(config, prompt_template, granularity)
    elif config.model_type == ModelType.JAN_LOCAL:
        return JanLocalClaimGenerator(config, prompt_template, granularity)
    elif config.model_type == ModelType.KG_TO_CLAIMS:
        return KGToClaimsGenerator(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
