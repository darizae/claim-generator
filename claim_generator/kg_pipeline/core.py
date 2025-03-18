import re
import json
import uuid
from dataclasses import dataclass
from typing import List, Optional, Callable

from .config import ModelConfig, KGModelType
from .models import OpenAIKGModel, BaseKGModel
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
    Minimal local parser that only supports the OpenAI backend.
    """

    def __init__(self,
                 model_config: ModelConfig,
                 kg_prompt_builder: Optional[Callable[[str], str]] = None):
        self.model_config = model_config
        if self.model_config.model_type != KGModelType.OPENAI:
            raise ValueError("KGParser only supports KGModelType.OPENAI in this minimal version.")
        self.model = OpenAIKGModel(self.model_config)
        self.kg_prompt_builder = kg_prompt_builder

    def parse_batch(self, texts: List[str]) -> List[KGOutput]:
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
        results = []
        for text, output_str in zip(texts, raw_outputs):
            data = self._extract_json(output_str)

            triples = []
            for triple in data.get("triples", []):
                if isinstance(triple, list) and len(triple) == 3:
                    triples.append(KGTriple(*triple))
                else:
                    # fallback or skip
                    pass

            results.append(KGOutput(
                id=str(uuid.uuid4()),
                source_text=text,
                triples=triples
            ))
        return results

    def _extract_json(self, output_str: str) -> dict:
        # Attempt direct parse
        try:
            data = json.loads(output_str.strip())
            return data
        except json.JSONDecodeError:
            pass
        # Fallback: try regex
        json_matches = re.findall(r'\{.*?\}', output_str, re.DOTALL)
        for match in json_matches:
            try:
                return json.loads(match)
            except:
                continue
        # If all fails:
        return {"triples": []}

    def save_to_json(self, outputs: List[KGOutput], path, triple_format: str='list'):
        data = []
        for o in outputs:
            if triple_format == 'list':
                t_data = [[t.subject, t.predicate, t.object] for t in o.triples]
            else:
                t_data = [t.__dict__ for t in o.triples]

            data.append({
                "id": o.id,
                "source_text": o.source_text,
                "triples": t_data
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
