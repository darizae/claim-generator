"""
kg_pipeline package - local copy of the old kg_parser for OpenAI-based knowledge graph extraction.
"""
__version__ = "0.1.0"

from .core import KGParser, KGOutput, KGTriple
from .config import ModelConfig, KGModelType
from .prompt_templates import REFINED_CLAIM_PROMPT
