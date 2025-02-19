#!/usr/bin/env python3
"""
generate_rose_kg.py

Processes the RoSE dataset to produce KG-based claims. Key features:

1) For each entry, if "kg_based_claims" is missing or empty, it calls the KG→claims pipeline.
2) If "kg_based_claims" is present and non-empty, it skips re-generation.
3) Saves partial results (both main JSON and synthetic JSON) every BATCH_SAVE_EVERY entries.
4) If an entry has "kg_based_claims" but is missing from the synthetic dataset, it builds
   a partial synthetic entry without calling the LLM again.

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

    # We might need REFINED_CLAIM_PROMPT for building partial synthetic entries
    from kg_parser.prompt_templates import REFINED_CLAIM_PROMPT

    # -------------------------------------------------------------------------
    # 6) Iterate Over Entries & Decide Whether to Generate
    # -------------------------------------------------------------------------
    processed_count = 0

    for dataset_name, entries in rose_data_with_claims.items():
        for entry in entries:
            record_id = entry.get("record_id", None)
            if "reference" not in entry:
                # No text => skip
                continue

            text = entry["reference"]

            # Check if "kg_based_claims" is present and non-empty
            # i.e., "already has claims"

            is_the_claim = entry["record_id"] == "cnndm_test_202"

            if not is_the_claim:
                continue

            if "kg_based_claims" in entry:
                # If we do have "kg_based_claims", let's see if it's empty:
                if entry["kg_based_claims"] and len(entry["kg_based_claims"]) > 0:
                    already_has_claims = True
                else:
                    # It's an empty list => treat as no claims
                    already_has_claims = False
            else:
                already_has_claims = False

            if is_the_claim:
                already_has_claims = False

            # (B) If "kg_based_claims" are present & non-empty, skip LLM calls,
            #     build partial synthetic if not in synthetic file
            if already_has_claims:
                continue

            # (C) If we do NOT have claims at all or it's an empty list => re-run pipeline

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

            # Store the synthetic object

            synthetic_dataset.append(syn_obj)

            synthetic_index[record_id] = syn_obj

            # Save synthetic data

            with synthetic_out_path.open("w", encoding="utf-8") as f_syn:

                json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

            # (D) Partial saves

        processed_count += 1

        if processed_count % BATCH_SAVE_EVERY == 0:
            print(f"[INFO] Processed {processed_count} entries => saving partial results...")

            # Save main data

            with out_path.open("w", encoding="utf-8") as f_save:
                json.dump(rose_data_with_claims, f_save, indent=2, ensure_ascii=False)


    # -------------------------------------------------------------------------
    # 7) Final save after the loop
    # -------------------------------------------------------------------------
    print(f"[INFO] Completed. Processed a total of {processed_count} entries in this run.")
    print(f"[INFO] Saving final results to {out_path}")
    with out_path.open("w", encoding="utf-8") as f_final:
        json.dump(rose_data_with_claims, f_final, indent=2, ensure_ascii=False)

    print(f"[INFO] Saving final synthetic data to {synthetic_out_path}")
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    print(f"[INFO] Done!")


if __name__ == "__main__":
    main()
