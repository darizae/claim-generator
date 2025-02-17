"""
generate_rose_kg.py

This script processes a RoSE dataset by:
  1) Parsing each entry's "reference" text into a KG.
  2) Converting each KG triple to a natural language claim.
  3) Saving, for each entry, a synthetic structure containing:
      - The original reference text.
      - The full kg-parser prompt.
      - The kg-parser output (as dict).
      - The full triple-to-claim prompts.
      - The array of claims.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from claim_generator import (
    ModelConfig,
    ModelType,
    create_generator,
    PromptTemplate,
)

# Configuration
DATASET_FILE = "rose_datasets.json"
OUTPUT_DIR = Path("data") / "outputs"
BATCH_SAVE_EVERY = 10
PRINT_PROGRESS = True

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def main():
    # 1) Load the dataset
    input_path = Path("data") / "inputs" / DATASET_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input dataset at {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        rose_data = json.load(f)

    # 2) Build or reload our output structure.
    out_filename = DATASET_FILE.replace(".json", "") + "_with_claims.json"
    out_path = OUTPUT_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[INFO] Found existing {out_path}. Loading to preserve previous runs.")
        with out_path.open("r", encoding="utf-8") as f_out:
            rose_data_with_claims = json.load(f_out)
    else:
        print(f"[INFO] No existing {out_path}. Creating a new output file.")
        rose_data_with_claims = rose_data  # assuming same structure

    # 3) Create a KG→Claims generator (using our new pipeline).
    config = ModelConfig(
        model_type=ModelType.KG_TO_CLAIMS,
        model_name_or_path="gpt-3.5-turbo",
        api_key=openai_api_key,
    )
    generator = create_generator(config, PromptTemplate.DEFAULT)

    synthetic_dataset = []
    processed_count = 0

    for dataset_name, entries in rose_data_with_claims.items():
        for entry in entries:
            if "reference" not in entry:
                raise ValueError(f"Entry {entry} is missing a reference.")
            text = entry["reference"]

            synthetic_entry, claims_list = generator.generate_claims_with_intermediates([text])
            synthetic_entry = synthetic_entry[0]
            entry["kg_based_claims"] = claims_list[0]
            synthetic_dataset.append(synthetic_entry)

            processed_count += 1
            if processed_count % BATCH_SAVE_EVERY == 0:
                print(f"[INFO] Processed {processed_count} entries; saving partial results...")
                with out_path.open("w", encoding="utf-8") as f_save:
                    json.dump(rose_data_with_claims, f_save, indent=2, ensure_ascii=False)
            if PRINT_PROGRESS:
                record_id = entry.get("record_id", "???")
                print(f"[INFO] Processed record {record_id}")

    print(f"[INFO] Completed processing. Saving final results to {out_path}")
    with out_path.open("w", encoding="utf-8") as f_final:
        json.dump(rose_data_with_claims, f_final, indent=2, ensure_ascii=False)

    # Save synthetic dataset.
    synthetic_out_filename = DATASET_FILE.replace(".json", "") + "_kg_claims_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote synthetic data to {synthetic_out_path}")


if __name__ == "__main__":
    main()
