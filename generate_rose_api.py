#!/usr/bin/env python3
"""
accumulate_rose_api.py

This script ensures that multiple runs append new claims to the SAME output JSON
(e.g., rose_datasets_small_with_claims.json), preserving prior runs' results.
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


SELECTED_MODEL = ModelChoice.GPT
SELECTED_DATASET = DatasetChoice.SMALL
SELECTED_PROMPT = PromptStrategy.MAXIMIZE_COVERAGE
BATCH_SAVE_EVERY = 10
PRINT_PROGRESS = True

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


###############################################################################
# 2) Main Script
###############################################################################

def main():
    # -------------------------------------------------------------------------
    # 2.1) Load existing "with_claims" or original JSON
    # -------------------------------------------------------------------------
    input_filename = SELECTED_DATASET.value
    input_path = Path("data") / "inputs" / input_filename

    # We'll call our cumulative output file "<original>_with_claims.json"
    out_filename = input_filename.replace(".json", "") + "_with_claims.json"
    out_path = OUTPUT_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        # File with claims already exists => load it so we can preserve old keys
        print(f"[INFO] Found existing {out_path}. Loading it to preserve prior runs.")
        with out_path.open("r", encoding="utf-8") as f:
            rose_data = json.load(f)
    else:
        # No existing file => load the original “clean” JSON
        print(f"[INFO] No existing {out_path}. Loading {input_path} as base.")
        with input_path.open("r", encoding="utf-8") as f:
            rose_data = json.load(f)

    # -------------------------------------------------------------------------
    # 2.2) Create generator
    # -------------------------------------------------------------------------
    config = get_model_config(SELECTED_MODEL)
    prompt_template_enum, template_kwargs = get_prompt_enum_and_kwargs(SELECTED_PROMPT)
    granularity = template_kwargs.get("granularity", None)
    generator = create_generator(config, prompt_template_enum, granularity=granularity)

    # Build a label for the newly generated claims, e.g. "gpt_maximize_coverage"
    model_label = "gpt" if SELECTED_MODEL == ModelChoice.GPT else "jan"
    prompt_label = SELECTED_PROMPT.name.lower()  # e.g. "maximize_coverage"
    joined_label = f"{model_label}_{prompt_label}"

    # -------------------------------------------------------------------------
    # 2.3) Synthetic dataset for debugging
    # -------------------------------------------------------------------------
    synthetic_dataset = []

    # -------------------------------------------------------------------------
    # 2.4) Generate claims for every entry's "reference"
    # -------------------------------------------------------------------------

    processed_count = 0

    for dataset_name, entries in rose_data.items():
        for entry in entries:
            # If "reference" doesn't exist, skip
            if "reference" not in entry:
                continue

            text_for_claims = entry["reference"]

            # Build final prompt for synthetic logging
            from claim_generator.prompts import get_prompt_template
            if prompt_template_enum == PromptTemplate.GRANULARITY and granularity:
                raw_prompt = get_prompt_template(prompt_template_enum, SOURCE_TEXT=text_for_claims,
                                                 granularity=granularity)
            else:
                raw_prompt = get_prompt_template(prompt_template_enum, SOURCE_TEXT=text_for_claims)

            # Generate
            result_list_of_lists = generator.generate_claims([text_for_claims])
            new_claims = result_list_of_lists[0] if result_list_of_lists else []

            # Attach to the same entry under e.g. "gpt_maximize_coverage"
            entry[joined_label] = new_claims

            # Synthetic info
            synthetic_dataset.append({
                "dataset_name": dataset_name,
                "record_id": entry.get("record_id", ""),
                "original_text": text_for_claims,
                "prompt_used": raw_prompt,
                "model_output_claims": new_claims
            })

            processed_count += 1
            if processed_count % BATCH_SAVE_EVERY == 0:
                print(f"[INFO] Processed {processed_count} entries so far; saving partial results...")
                with out_path.open("w", encoding="utf-8") as f_out:
                    json.dump(rose_data, f_out, indent=2, ensure_ascii=False)

            if not PRINT_PROGRESS:
                continue

            record_id = entry["record_id"]
            print(f"[INFO] Generated claims for record {record_id}")

    # -------------------------------------------------------------------------
    # 2.5) Write out the updated "with_claims" JSON again
    # -------------------------------------------------------------------------
    print(f"[INFO] Completed processing. Saving final results to {out_path}.")
    with out_path.open("w", encoding="utf-8") as f_out:
        json.dump(rose_data, f_out, indent=2, ensure_ascii=False)

    print(f"[INFO] Updated {out_path} with new claims under key '{joined_label}'.")

    # Also write out the synthetic data in a separate file

    if SELECTED_DATASET == DatasetChoice.SMALL:
        return

    synthetic_out_filename = input_filename.replace(".json", "") + f"__{joined_label}_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename
    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote synthetic data => {synthetic_out_path}")


if __name__ == "__main__":
    main()
