#!/usr/bin/env python3
"""
generate_rose_kg.py

Processes the RoSE dataset to produce KG-based claims. Key features:

1) Skips API calls if "kg_based_claims" already present.
2) Saves partial results (both main JSON and synthetic JSON) every 10 entries.
3) If an entry has existing "kg_based_claims" but no synthetic data yet,
   it builds a partial synthetic entry without calling the LLM.

NOTE: This script assumes each entry has a unique "record_id" to identify it
      for synthetic data. If not, you'll need to adapt the uniqueness logic.
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

# We'll reuse the KGToClaimsGenerator for actual generation when needed.
# That class' method `generate_claims_with_intermediates` can provide:
#  - reference text
#  - kg_parser_prompt
#  - kg_output
#  - triple_to_claim_prompts
#  - claims

# 1) Configuration
DATASET_FILE = "rose_datasets.json"  # or "rose_datasets.json"
OUTPUT_DIR = Path("data") / "outputs"
BATCH_SAVE_EVERY = 200
PRINT_PROGRESS = True

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def main():
    # -------------------------------------------------------------------------
    # 2) Load the base dataset
    # -------------------------------------------------------------------------
    input_path = Path("data") / "inputs" / DATASET_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input dataset at {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        rose_data = json.load(f)

    # -------------------------------------------------------------------------
    # 3) Prepare/Load the "with_claims" output
    # -------------------------------------------------------------------------
    out_filename = DATASET_FILE.replace(".json", "") + "_with_claims.json"
    out_path = OUTPUT_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[INFO] Found existing '{out_path}'. Loading it to preserve previous runs.")
        with out_path.open("r", encoding="utf-8") as f_out:
            rose_data_with_claims = json.load(f_out)
    else:
        print(f"[INFO] No existing '{out_path}'. We'll create it fresh.")
        rose_data_with_claims = rose_data  # assume same structure (dict of lists)

    # -------------------------------------------------------------------------
    # 4) Prepare/Load the synthetic dataset for partial saves
    # -------------------------------------------------------------------------
    synthetic_out_filename = DATASET_FILE.replace(".json", "") + "_kg_claims_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename
    synthetic_dataset = []
    synthetic_index = {}  # map record_id -> synthetic object

    if synthetic_out_path.exists():
        print(f"[INFO] Found existing synthetic file '{synthetic_out_path}'. Loading it.")
        with synthetic_out_path.open("r", encoding="utf-8") as f_syn:
            existing_synthetic = json.load(f_syn)
        # Build an index so we don't duplicate or overwrite
        for syn_entry in existing_synthetic:
            rec_id = syn_entry.get("record_id")
            if rec_id is not None:
                synthetic_index[rec_id] = syn_entry
        synthetic_dataset = existing_synthetic
    else:
        print(f"[INFO] No existing synthetic data file '{synthetic_out_path}'. Starting empty.")
        synthetic_dataset = []
        synthetic_index = {}

    # -------------------------------------------------------------------------
    # 5) Create the KG->Claims generator
    # -------------------------------------------------------------------------
    config = ModelConfig(
        model_type=ModelType.KG_TO_CLAIMS,
        model_name_or_path="gpt-3.5-turbo",  # or your desired LLM
        api_key=openai_api_key,
        temperature=0.0,
    )
    generator = create_generator(config, PromptTemplate.DEFAULT)

    # -------------------------------------------------------------------------
    # 6) Iteration & partial logic
    # -------------------------------------------------------------------------
    from kg_parser.prompt_templates import REFINED_CLAIM_PROMPT  # for reconstructing prompt

    processed_count = 0

    for dataset_name, entries in rose_data_with_claims.items():
        for entry in entries:
            record_id = entry.get("record_id", None)
            if "reference" not in entry:
                continue

            text = entry["reference"]
            already_has_claims = "kg_based_claims" in entry

            # (A) If we've already built synthetic data for this record, skip entirely
            if record_id in synthetic_index:
                # It's fully accounted for in the synthetic dataset
                if PRINT_PROGRESS:
                    print(f"[INFO] record_id={record_id} => already in synthetic dataset. Skipping.")
                continue

            if already_has_claims:
                # ---------------------------------------------------------
                # (B) We already have claims => skip API calls,
                # but build a partial synthetic entry if it doesn't exist.
                # ---------------------------------------------------------
                if PRINT_PROGRESS:
                    print(
                        f"[INFO] record_id={record_id} => claims already present. Building synthetic from existing data...")

                # Build partial synthetic data. We do NOT have the actual triples or triple prompts
                # unless we previously stored them. So we fill placeholders:
                claims_from_main = entry["kg_based_claims"]

                synthetic_entry = {
                    "record_id": record_id,
                    "dataset_name": dataset_name,
                    "reference": text,
                    # We can still reconstruct the parser prompt for reference:
                    "kg_parser_prompt": REFINED_CLAIM_PROMPT.format(input=text),
                    # But we never stored the triple data => set placeholders:
                    "kg_output": "unavailable (already generated; no stored triple data)",
                    "triple_to_claim_prompts": "unavailable (already generated; no stored triple data)",
                    "claims": claims_from_main
                }

                synthetic_dataset.append(synthetic_entry)
                synthetic_index[record_id] = synthetic_entry

            else:
                # ---------------------------------------------------------
                # (C) We do NOT have claims => we need to call the pipeline.
                # ---------------------------------------------------------
                if PRINT_PROGRESS:
                    print(f"[INFO] record_id={record_id} => generating claims via KG parser + triple->claim...")

                # We'll generate both the final claims and the synthetic info:
                synthetic_entries, all_claims_lists = generator.generate_claims_with_intermediates([text])
                # Each is a single-element list because we're passing a single text
                syn_obj = synthetic_entries[0]  # dict with references, prompts, etc.
                claims = all_claims_lists[0]

                # Add "record_id" and "dataset_name" for clarity
                syn_obj["record_id"] = record_id
                syn_obj["dataset_name"] = dataset_name

                # Save final claims in the main data structure
                entry["kg_based_claims"] = claims

                # Add synthetic object to our in-memory collection
                synthetic_dataset.append(syn_obj)
                synthetic_index[record_id] = syn_obj

            # (D) Either way, we increment the processed count for partial saving
            processed_count += 1
            if processed_count % BATCH_SAVE_EVERY == 0:
                print(f"[INFO] Processed {processed_count} entries => saving partial results...")

                # Save the main data
                with out_path.open("w", encoding="utf-8") as f_save:
                    json.dump(rose_data_with_claims, f_save, indent=2, ensure_ascii=False)

                # Save the synthetic data
                with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
                    json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 7) Final save after the loop
    # -------------------------------------------------------------------------
    print(f"[INFO] Completed. Saving final results to {out_path}")
    with out_path.open("w", encoding="utf-8") as f_final:
        json.dump(rose_data_with_claims, f_final, indent=2, ensure_ascii=False)

    print(f"[INFO] Saving final synthetic data to {synthetic_out_path}")
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    print(f"[INFO] Done!")


if __name__ == "__main__":
    main()
