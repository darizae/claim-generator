from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# If you still want a device_selector, keep it; else remove or stub it out:
#from device_selector import check_or_select_device

class KGModelType(Enum):
    OPENAI = "openai"



@dataclass
class ModelConfig:
    """
    Holds model-related configuration for knowledge graph usage.
    """
    model_type: KGModelType
    model_name_or_path: str  # e.g., "gpt-3.5-turbo"
    api_key: str = None
    temperature: float = 0.1
    max_length: int = 512
    batch_size: int = 8

    # For our minimal example, we skip device, endpoint_url, etc.

    #def __post_init__(self):
    #    self.device = check_or_select_device(None)  # or remove if not needed
