"""
generate_rose_kg.py

Example script that:
  1) Loads a RoSE dataset from JSON
  2) Uses the KG_TO_CLAIMS pipeline
  3) Saves results (including intermediate KG, final claims, prompts, etc.)
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

# We assume you have: from .config import PathConfig, etc. if you need them.

# Choose your dataset, model, etc.
DATASET_FILE = "rose_datasets_small.json"  # or "rose_datasets.json"
OUTPUT_DIR = Path("data") / "outputs"
BATCH_SAVE_EVERY = 10
PRINT_PROGRESS = True

# Load environment for API keys, if needed
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

    # If the file already exists, we load so we can append. Otherwise, start fresh.
    if out_path.exists():
        print(f"[INFO] Found existing {out_path}. Loading to preserve previous runs.")
        with out_path.open("r", encoding="utf-8") as f_out:
            rose_data_with_claims = json.load(f_out)
    else:
        print(f"[INFO] No existing {out_path}. We'll create it fresh.")
        rose_data_with_claims = rose_data  # structure is presumably the same shape

    # 3) Create a KG->Claims generator (with chosen backend)
    config = ModelConfig(
        model_type=ModelType.KG_TO_CLAIMS,
        model_name_or_path="gpt-3.5-turbo",  # or "openai/whisper" or "jan_local" or HF name
        api_key=openai_api_key,
    )
    generator = create_generator(config, PromptTemplate.DEFAULT)

    synthetic_dataset = []

    # 4) For each entry, run the pipeline => final claims
    processed_count = 0
    for dataset_name, entries in rose_data_with_claims.items():
        for entry in entries:
            if "reference" not in entry:
                continue
            text = entry["reference"]

            # (a) Generate
            list_of_claims_lists = generator.generate_claims([text])
            claims_for_this_text = list_of_claims_lists[0]

            # (b) Save to an appropriate key in that JSON
            # e.g., "kg_claims" or "kg_based_claims" or however you'd like to store it
            entry["kg_based_claims"] = claims_for_this_text

            # (c) For “synthetic” logging, gather the parse results:
            # The KGToClaimsGenerator actually used self.kg_parser inside,
            # so let's do an extra parse call just to retrieve the raw KG data,
            # or you can modify KGToClaimsGenerator to expose that intermediate.
            # For simplicity, we'll do an inline parse:
            kg_outputs = generator.kg_parser.parse_batch([text])
            if kg_outputs:
                # Just one item
                kg_out = kg_outputs[0]
                # We'll build a triple->claim mapping
                triple_claim_map = []
                for triple, claim_str in zip(kg_out.triples, claims_for_this_text):
                    # We'll store the prompt used if we want. For the triple->claim
                    # we used the TRIPLE_TO_CLAIM_PROMPT. We'll store it explicitly:
                    triple_prompt = f"[{triple.subject}, {triple.predicate}, {triple.object}]"
                    triple_claim_map.append({
                        "triple": [triple.subject, triple.predicate, triple.object],
                        "prompt_used": triple_prompt,
                        "claim_output": claim_str
                    })
                # Add it to a synthetic record
                synthetic_dataset.append({
                    "dataset_name": dataset_name,
                    "record_id": entry.get("record_id", ""),
                    "original_text": text,
                    "kg_json": [{
                        "triples": [
                            [t.subject, t.predicate, t.object] for t in kg_out.triples
                        ]
                    }],
                    "triple_to_claim": triple_claim_map
                })

            processed_count += 1
            if processed_count % BATCH_SAVE_EVERY == 0:
                print(f"[INFO] Processed {processed_count} entries => partial save.")
                with out_path.open("w", encoding="utf-8") as f_save:
                    json.dump(rose_data_with_claims, f_save, indent=2, ensure_ascii=False)

            if PRINT_PROGRESS:
                print(f"[INFO] Generated {len(claims_for_this_text)} claims for record {entry.get('record_id', '???')}")

    # 5) Final save
    print(f"[INFO] Completed. Saving final results to {out_path}")
    with out_path.open("w", encoding="utf-8") as f_final:
        json.dump(rose_data_with_claims, f_final, indent=2, ensure_ascii=False)

    # 6) Also store the “synthetic dataset” with prompts, KG, etc.
    synthetic_out_filename = DATASET_FILE.replace(".json", "") + "_kg_claims_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote synthetic data to {synthetic_out_path}")


if __name__ == "__main__":
    main()
