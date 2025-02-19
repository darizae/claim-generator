#!/usr/bin/env python3
"""
generate_rose_api_missing.py

Similar to generate_rose_api.py, but ONLY generates new claims for entries where
the desired strategy’s output is missing or empty. Also updates the corresponding
synthetic dataset JSON file with the newly generated content.

Example usage:
    - Adjust the manual constants below (SELECTED_MODEL, SELECTED_DATASET, etc.).
    - This script loads the existing "with_claims" file and the matching synthetic dataset file,
      checks if `entry[joined_label]` is missing or empty,
      and if so, it calls the model to generate claims.
    - Logs the record_id so you can diagnose which entries were updated.
    - Updates or inserts into the synthetic dataset, then saves partial results.
"""

import os
import json
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from claim_generator import (
    ModelConfig,
    ModelType,
    create_generator,
    PromptTemplate,
)


###############################################################################
# 1) Enums and Config
###############################################################################

class ModelChoice(Enum):
    GPT = "gpt-3.5-turbo"
    JAN = "jan_local"


class DatasetChoice(Enum):
    SMALL = "rose_datasets_small.json"
    FULL = "rose_datasets.json"


class PromptStrategy(Enum):
    DEFAULT = "DEFAULT"
    MAXIMIZE_ATOMICITY = "MAXIMIZE_ATOMICITY"
    MAXIMIZE_COVERAGE = "MAXIMIZE_COVERAGE"
    GRANULARITY_LOW = "GRANULARITY_LOW"
    GRANULARITY_MEDIUM = "GRANULARITY_MEDIUM"
    GRANULARITY_HIGH = "GRANULARITY_HIGH"


# -------------------- MANUAL CONFIG CONSTANTS --------------------
SELECTED_MODEL = ModelChoice.GPT
SELECTED_DATASET = DatasetChoice.FULL
SELECTED_PROMPT = PromptStrategy.MAXIMIZE_COVERAGE

BATCH_SAVE_EVERY = 1
PRINT_PROGRESS = True
# ----------------------------------------------------------------

OUTPUT_DIR = Path("data") / "outputs"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_model_config(model_choice: ModelChoice) -> ModelConfig:
    if model_choice == ModelChoice.GPT:
        return ModelConfig(
            model_type=ModelType.OPENAI,
            model_name_or_path=model_choice.value,
            api_key=openai_api_key,
        )
    elif model_choice == ModelChoice.JAN:
        return ModelConfig(
            model_type=ModelType.JAN_LOCAL,
            model_name_or_path="llama3.2-1b-instruct",
            endpoint_url="http://localhost:1337/v1/chat/completions",
        )
    else:
        raise ValueError(f"Unhandled model choice: {model_choice}")


def get_prompt_enum_and_kwargs(strategy: PromptStrategy):
    if strategy == PromptStrategy.DEFAULT:
        return (PromptTemplate.DEFAULT, {})
    elif strategy == PromptStrategy.MAXIMIZE_ATOMICITY:
        return (PromptTemplate.MAXIMIZE_ATOMICITY, {})
    elif strategy == PromptStrategy.MAXIMIZE_COVERAGE:
        return (PromptTemplate.MAXIMIZE_COVERAGE, {})
    elif strategy == PromptStrategy.GRANULARITY_LOW:
        return (PromptTemplate.GRANULARITY, {"granularity": "low"})
    elif strategy == PromptStrategy.GRANULARITY_MEDIUM:
        return (PromptTemplate.GRANULARITY, {"granularity": "medium"})
    elif strategy == PromptStrategy.GRANULARITY_HIGH:
        return (PromptTemplate.GRANULARITY, {"granularity": "high"})
    else:
        raise ValueError(f"Unhandled prompt strategy: {strategy}")


def main():
    # -------------------------------------------------------------------------
    # 1) Determine input + output for main dataset
    #    We read from the existing *with_claims* file, not the original,
    #    because we want to fill in the missing parts.
    # -------------------------------------------------------------------------
    input_filename = SELECTED_DATASET.value  # e.g. "rose_datasets.json"
    base_name = input_filename.replace(".json", "")  # e.g. "rose_datasets"
    out_filename = base_name + "_with_claims.json"   # e.g. "rose_datasets_with_claims_with_click.json"

    out_path = OUTPUT_DIR / out_filename
    if not out_path.exists():
        print(f"[ERROR] The file '{out_path}' does not exist. You must run generate_rose_api.py first or ensure it exists.")
        return

    print(f"[INFO] Loading existing 'with_claims' data from '{out_path}'...")
    with out_path.open("r", encoding="utf-8") as f_in:
        rose_data = json.load(f_in)

    # -------------------------------------------------------------------------
    # 2) Determine synthetic dataset file for the chosen model & strategy
    # -------------------------------------------------------------------------
    model_label = "gpt" if SELECTED_MODEL == ModelChoice.GPT else "jan"
    prompt_label = SELECTED_PROMPT.name.lower()
    joined_label = f"{model_label}_{prompt_label}"

    synthetic_out_filename = base_name + f"__{joined_label}_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename

    # Load or create the synthetic dataset
    synthetic_dataset = []
    synthetic_index = {}

    if synthetic_out_path.exists():
        print(f"[INFO] Found existing synthetic file '{synthetic_out_path}'. Loading it.")
        with synthetic_out_path.open("r", encoding="utf-8") as f_syn:
            existing_synthetic = json.load(f_syn)
        # Build an index by record_id
        for obj in existing_synthetic:
            rec_id = obj.get("record_id")
            if rec_id is not None:
                synthetic_index[rec_id] = obj
        synthetic_dataset = existing_synthetic
    else:
        print(f"[INFO] No existing synthetic file '{synthetic_out_path}'. We'll create it fresh.")
        synthetic_dataset = []
        synthetic_index = {}

    # -------------------------------------------------------------------------
    # 3) Create the generator
    # -------------------------------------------------------------------------
    config = get_model_config(SELECTED_MODEL)
    prompt_template_enum, template_kwargs = get_prompt_enum_and_kwargs(SELECTED_PROMPT)
    granularity = template_kwargs.get("granularity", None)

    generator = create_generator(config, prompt_template_enum, granularity=granularity)

    from claim_generator.prompts import get_prompt_template

    print(f"[INFO] We will only re-generate claims where '{joined_label}' is missing or empty.")

    processed_count = 0
    updated_count = 0

    # -------------------------------------------------------------------------
    # 4) Iterate over the "with_claims" data, generate only if empty
    # -------------------------------------------------------------------------
    for dataset_name, entries in rose_data.items():
        if not isinstance(entries, list):
            # skip if structure is unexpected
            continue

        for entry in entries:
            # Must have reference text to generate claims
            if "reference" not in entry:
                continue

            record_id = entry.get("record_id", "NO_RECORD_ID")
            text_for_claims = entry["reference"]
            existing_claims = entry.get(joined_label, None)

            # If the key does not exist or is empty => re-generate
            if existing_claims and len(existing_claims) > 0:
                # Already populated => skip
                processed_count += 1
                continue

            # Build the actual prompt text (for debugging or synthetic logs)
            if prompt_template_enum == PromptTemplate.GRANULARITY and granularity:
                raw_prompt = get_prompt_template(prompt_template_enum, SOURCE_TEXT=text_for_claims,
                                                 granularity=granularity)
            else:
                raw_prompt = get_prompt_template(prompt_template_enum, SOURCE_TEXT=text_for_claims)

            # Re-generate
            print(f"[INFO] Re-generating claims for dataset='{dataset_name}' record_id='{record_id}'...")
            result_list_of_lists = generator.generate_claims([text_for_claims])
            new_claims = result_list_of_lists[0] if result_list_of_lists else []

            # Store in the main data
            entry[joined_label] = new_claims

            # Also update synthetic dataset
            if record_id in synthetic_index:
                # update existing object
                syn_obj = synthetic_index[record_id]
                syn_obj["dataset_name"] = dataset_name
                syn_obj["original_text"] = text_for_claims
                syn_obj["prompt_used"] = raw_prompt
                syn_obj["model_output_claims"] = new_claims
            else:
                # create new object
                syn_obj = {
                    "record_id": record_id,
                    "dataset_name": dataset_name,
                    "original_text": text_for_claims,
                    "prompt_used": raw_prompt,
                    "model_output_claims": new_claims
                }
                synthetic_dataset.append(syn_obj)
                synthetic_index[record_id] = syn_obj

            print(f"  -> Generated {len(new_claims)} claims for record_id='{record_id}'.")
            updated_count += 1
            processed_count += 1

            # Partial save
            if updated_count % BATCH_SAVE_EVERY == 0:
                print(f"[INFO] Re-generated claims for {updated_count} records so far; saving partial results...")
                # Save main
                with out_path.open("w", encoding="utf-8") as f_out:
                    json.dump(rose_data, f_out, indent=2, ensure_ascii=False)

                # Save synthetic
                with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
                    json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 5) Save final "with_claims" after re-generation
    # -------------------------------------------------------------------------
    print(f"[INFO] Done. Processed {processed_count} entries. Re-generated for {updated_count} entries.")
    print(f"[INFO] Saving final updated dataset to {out_path} ...")

    with out_path.open("w", encoding="utf-8") as f_out:
        json.dump(rose_data, f_out, indent=2, ensure_ascii=False)

    # Save synthetic as well
    print(f"[INFO] Saving final synthetic dataset to {synthetic_out_path} ...")
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    print(f"[INFO] Updated '{out_path}' and '{synthetic_out_path}' with newly generated claims for '{joined_label}'.")
    print("[INFO] Script complete.")


if __name__ == "__main__":
    main()
