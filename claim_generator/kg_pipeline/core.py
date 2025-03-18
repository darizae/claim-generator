import re
import json
import uuid
from dataclasses import dataclass
from typing import List, Optional, Callable
from pathlib import Path

from .config import ModelConfig, KGModelType
from .models import (
    HuggingFaceKGModel,
    OpenAIKGModel,
    JanLocalKGModel,
)
from .prompt_templates import REFINED_CLAIM_PROMPT


@dataclass
class KGTriple:
    subject: str
    predicate: str
    object: str


@dataclass
class KGOutput:
    id: str
    source_text: str
    triples: List[KGTriple]


class KGParser:
    """
    Knowledge Graph parser that supports OpenAI, HuggingFace, and optionally local Jan server usage.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        kg_prompt_builder: Optional[Callable[[str], str]] = None
    ):
        self.model_config = model_config
        self.model = self._initialize_model()
        self.kg_prompt_builder = kg_prompt_builder

    def _initialize_model(self):
        if self.model_config.model_type == KGModelType.HUGGINGFACE:
            return HuggingFaceKGModel(self.model_config)
        elif self.model_config.model_type == KGModelType.OPENAI:
            return OpenAIKGModel(self.model_config)
        elif self.model_config.model_type == KGModelType.JAN_LOCAL:
            return JanLocalKGModel(self.model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_config.model_type}")

    def parse_batch(self, texts: List[str]) -> List[KGOutput]:
        """
        Build prompts for each text (or use a custom builder), feed them in batches,
        and parse out final structured results.
        """
        raw_prompts = []
        for t in texts:
            if self.kg_prompt_builder:
                prompt = self.kg_prompt_builder(t)
            else:
                prompt = REFINED_CLAIM_PROMPT.format(input=t)
            raw_prompts.append(prompt)

        raw_outputs = self.model.generate_kg_prompts(raw_prompts)
        return self._process_outputs(texts, raw_outputs)

    def _process_outputs(self, texts: List[str], raw_outputs: List[str]) -> List[KGOutput]:
        """
        Attempt to parse each output string as valid JSON containing "triples".
        """
        results = []
        for text, output_str in zip(texts, raw_outputs):
            data = None
            # Try direct parse:
            try:
                data = json.loads(output_str.strip())
            except json.JSONDecodeError:
                # If direct parse fails, attempt to find JSON via regex
                json_matches = re.findall(r'\{.*?\}', output_str, re.DOTALL)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                # If we fail to parse any JSON, fallback to empty
                data = {"triples": []}

            triples = []
            for triple in data.get("triples", []):
                if isinstance(triple, list) and len(triple) == 3:
                    triples.append(KGTriple(*triple))
                elif isinstance(triple, dict):
                    # If it's a dict with subject/predicate/object keys
                    sub = triple.get("subject", "")
                    pred = triple.get("predicate", "")
                    obj = triple.get("object", "")
                    if sub and pred and obj:
                        triples.append(KGTriple(sub, pred, obj))

            results.append(KGOutput(
                id=str(uuid.uuid4()),
                source_text=text,
                triples=triples
            ))
        return results

    def save_to_json(self, outputs: List[KGOutput], path: Path, triple_format: str = 'list'):
        """
        Saves the list of KGOutput to a JSON file with either 'list' or 'dict' triple format.
        """
        if not isinstance(path, Path):
            path = Path(path)

        data = []
        for o in outputs:
            if triple_format == 'list':
                t_data = [[t.subject, t.predicate, t.object] for t in o.triples]
            elif triple_format == 'dict':
                t_data = [vars(t) for t in o.triples]
            else:
                raise ValueError(f"Unknown triple format: {triple_format}")

            data.append({
                "id": o.id,
                "source_text": o.source_text,
                "triples": t_data
            })

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
