from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# Optional device selector stub (replace with your own logic or remove if not needed).
def check_or_select_device(requested_device: str = None) -> str:
    """
    Simple example of picking a device.
    Replace with your actual logic or remove entirely if you only run on CPU/GPU known environment.
    """
    if requested_device:
        return requested_device
    # Attempt GPU if available, else fallback CPU:
    # (In a real project you might check torch.cuda.is_available() or similar.)
    return "cpu"


class KGModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    JAN_LOCAL = "jan_local"  # Optional if you want to keep local endpoints


@dataclass
class ModelConfig:
    """
    Holds model-related configuration for knowledge graph usage.
    """
    model_type: KGModelType
    model_name_or_path: str  # e.g. "gpt-3.5-turbo" or "meta-llama/Llama-3.3-70B-Instruct"
    device: str = None       # e.g. "cpu" or "cuda". If None, we attempt auto-detection.
    api_key: str = None      # For OpenAI usage
    endpoint_url: str = None # For Jan local usage, if you keep that
    temperature: float = 0.1
    max_length: int = 512
    batch_size: int = 8

    def __post_init__(self):
        # If you want to auto-select device for HuggingFace:
        self.device = check_or_select_device(self.device)
