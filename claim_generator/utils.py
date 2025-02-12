import json
from pathlib import Path
from typing import List, Dict, Any


def save_claims_to_json(
        items: List[Dict[str, Any]],
        output_path: Path
) -> None:
    """
    Save the items (with claims) to the specified JSON path.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
