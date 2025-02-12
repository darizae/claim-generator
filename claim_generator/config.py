from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from device_selector import check_or_select_device


class ModelType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    JAN_LOCAL = "jan_local"
    KG_TO_CLAIMS = "kg_to_claims"
    # If needed, you can add more model types in the future.


@dataclass
class PathConfig:
    """
    Example path configuration.
    In a real project, you might customize or add more as needed.
    """
    base_dir: Path = Path(__file__).resolve().parent.parent
    input_dir: Path = base_dir / "data" / "inputs"
    output_dir: Path = base_dir / "data" / "outputs"


@dataclass
class ModelConfig:
    """
    Configuration for the model used to generate claims.
    """
    model_type: ModelType
    model_name_or_path: str  # e.g., "gpt-3.5-turbo" or "facebook/bart-large"
    device: str = None  # Will be auto-selected if None
    api_key: str = None  # e.g., OpenAI or other
    endpoint_url: str = None  # e.g., for JanLocal usage
    temperature: float = 0.0
    max_length: int = 512
    batch_size: int = 8

    def __post_init__(self):
        # Auto-select device if not specified
        self.device = check_or_select_device(self.device)
