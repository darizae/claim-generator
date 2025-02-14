#!/usr/bin/env python3
"""
generate_rose_api.py

Script to generate claims for RoSE dataset entries using different model backends
(GPT-3.5 vs Jan) and different prompt strategies (e.g., default, atomicity, coverage, etc.).

Usage:
  1. Manually update the flags near the top to pick model/dataset/prompt style.
  2. Make sure you have a .env file with OPENAI_API_KEY (if using GPT-3.5).
  3. Run: python generate_rose_api.py
  4. Check the outputs in data/outputs or whichever path you choose.

Requires:
  - pip install python-dotenv
  - The claim_generator package (your local module).
  - A valid .env file if using OpenAI GPT.

Saves:
  - A JSON file that merges the newly generated claims into the original input structure.
  - A "synthetic" JSON file with prompt + model output for each text.

"""

import os
import json
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Import from your claim_generator module
from claim_generator import (
    ModelConfig,
    ModelType,
    create_generator,
    PromptTemplate,
    get_prompt_template
)


# ---------------------------------------------------------------------------
# 1) Define flags to control the script’s behavior
# ---------------------------------------------------------------------------

class ModelChoice(Enum):
    GPT = "gpt-3.5-turbo"  # Our GPT-based model
    JAN = "jan_local"  # Our Jan local model


class DatasetChoice(Enum):
    SMALL = "rose_datasets_small.json"
    FULL = "rose_datasets.json"


# You can extend prompt strategies if you like. We match them to PromptTemplate.
class PromptStrategy(Enum):
    DEFAULT = "DEFAULT"
    MAXIMIZE_ATOMICITY = "MAXIMIZE_ATOMICITY"
    MAXIMIZE_COVERAGE = "MAXIMIZE_COVERAGE"
    GRANULARITY_LOW = "GRANULARITY_LOW"
    GRANULARITY_MEDIUM = "GRANULARITY_MEDIUM"
    GRANULARITY_HIGH = "GRANULARITY_HIGH"
    # You could add a COVERAGE variant, etc.


# Manually tweak these as you wish:
SELECTED_MODEL = ModelChoice.JAN
SELECTED_DATASET = DatasetChoice.SMALL
SELECTED_PROMPT = PromptStrategy.MAXIMIZE_ATOMICITY

# Output sub-folder
OUTPUT_DIR = Path("data") / "outputs"

# ---------------------------------------------------------------------------
# 2) Prepare environment and model configs
# ---------------------------------------------------------------------------

load_dotenv()  # so OPENAI_API_KEY is loaded from .env
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_model_config(model_choice: ModelChoice):
    """
    Return an appropriate ModelConfig object depending on which model is selected.
    Adjust as needed for your real model paths.
    """
    if model_choice == ModelChoice.GPT:
        # GPT 3.5 using the OpenAI API
        return ModelConfig(
            model_type=ModelType.OPENAI,
            model_name_or_path=model_choice.value,  # "gpt-3.5-turbo"
            api_key=openai_api_key,
            # temperature=0.0,  # tweak as needed
            # max_length=512,
            # batch_size=1
        )
    elif model_choice == ModelChoice.JAN:
        # Example Jan local usage
        return ModelConfig(
            model_type=ModelType.JAN_LOCAL,
            model_name_or_path="llama3.2-1b-instruct",
            endpoint_url="http://localhost:1337/v1/chat/completions",
            # api_key=None,
            # temperature=0.7,
            # max_length=512,
            # batch_size=4
        )
    else:
        raise ValueError(f"Unhandled model choice: {model_choice}")


def get_prompt_enum_and_kwargs(strategy: PromptStrategy):
    """
    Map our custom strategy enum to (PromptTemplate, optional kwargs).
    For 'GRANULARITY', we pass an extra param called {granularity} to the template.
    """
    if strategy == PromptStrategy.DEFAULT:
        return (PromptTemplate.DEFAULT, {})
    elif strategy == PromptStrategy.MAXIMIZE_ATOMICITY:
        return (PromptTemplate.MAXIMIZE_ATOMICITY, {})
    elif strategy == PromptStrategy.MAXIMIZE_COVERAGE:
        return (PromptTemplate.MAXIMIZE_COVERAGE, {})
    elif strategy in (
            PromptStrategy.GRANULARITY_LOW,
            PromptStrategy.GRANULARITY_MEDIUM,
            PromptStrategy.GRANULARITY_HIGH
    ):
        # We use PromptTemplate.GRANULARITY from claim_generator/prompts.py
        # We'll pass a "granularity" param to it, e.g., "low", "medium", or "high".
        # We can parse from the strategy name or use the global GRANULARITY_LEVEL.
        if strategy == PromptStrategy.GRANULARITY_LOW:
            gran = "low"
        elif strategy == PromptStrategy.GRANULARITY_MEDIUM:
            gran = "medium"
        else:
            gran = "high"
        return (PromptTemplate.GRANULARITY, {"granularity": gran})
    else:
        raise ValueError(f"Unhandled prompt strategy: {strategy}")


# ---------------------------------------------------------------------------
# 3) Main logic
# ---------------------------------------------------------------------------

def main():
    # 3.1) Pick which input file to read
    input_filename = SELECTED_DATASET.value
    input_path = Path("data") / "inputs" / input_filename

    # 3.2) Prepare model config and claim generator
    config = get_model_config(SELECTED_MODEL)

    prompt_template_enum, template_kwargs = get_prompt_enum_and_kwargs(SELECTED_PROMPT)
    # Create the actual prompt generator with the code from claim_generator
    # (openAI or Jan usage).
    granularity = template_kwargs["granularity"] if template_kwargs.get("granularity") is not None else None
    generator = create_generator(config, prompt_template_enum, granularity=granularity)

    # 3.3) Load the entire RoSE dataset JSON
    with input_path.open("r", encoding="utf-8") as f:
        rose_data = json.load(f)

    # We will store a second structure for the “synthetic dataset”
    synthetic_dataset = []

    # 3.4) For each dataset (e.g., "cnndm_test", "xsum", "samsum" etc.)
    #      generate claims from each entry's "reference" field.

    # Construct a short label for the output keys (e.g. "GPT_DEFAULT", "JAN_ATOMIC", etc.)
    model_label = "gpt" if SELECTED_MODEL == ModelChoice.GPT else "jan"
    prompt_label = SELECTED_PROMPT.value.lower()
    joined_label = f"{model_label}_{prompt_label}"

    # If using granularity, add that to the label:
    if "granularity" in template_kwargs:
        gran_val = template_kwargs["granularity"]
        joined_label += f"_{gran_val}"

    # We'll process each array in the dictionary:
    for dataset_name, entries in rose_data.items():
        for entry in entries:
            # The text we want to generate claims for is in the "reference" key
            text_for_claims = entry["reference"]

            # Build the final prompt text ourselves (optional) just so we can record it in synthetic dataset
            # In your actual generator, the build_claim_extraction_prompt is done internally,
            # but for logging, we can do it manually:
            raw_prompt = None
            # If you want the EXACT final string used by the generator, you can do:
            if prompt_template_enum.name == "GRANULARITY":
                # get_prompt_template returns the final formatted string
                from claim_generator.prompts import get_prompt_template
                raw_prompt = get_prompt_template(prompt_template_enum, **template_kwargs, SOURCE_TEXT=text_for_claims)
            else:
                from claim_generator.prompts import get_prompt_template
                raw_prompt = get_prompt_template(prompt_template_enum, SOURCE_TEXT=text_for_claims)

            # In practice, calling generator.generate_claims on a single item or in a batch is up to you.
            # We'll do it on a single item for simplicity:
            result_list_of_lists = generator.generate_claims([text_for_claims])
            # result_list_of_lists is [[claims...]], so we want the first
            if not result_list_of_lists:
                # fallback
                new_claims = []
            else:
                new_claims = result_list_of_lists[0]

            # Store in the original data structure under a new key
            # e.g. "gpt_default" or "jan_maximize_atomicity"
            entry[joined_label] = new_claims

            # Also add an entry in our synthetic dataset
            synthetic_dataset.append({
                "dataset_name": dataset_name,
                "record_id": entry.get("record_id", ""),
                "original_text": text_for_claims,
                "prompt_used": raw_prompt,
                "model_output_claims": new_claims
            })

    # 3.5) Save the updated RoSE data back out
    # e.g., "rose_datasets_small__gpt_default.json"
    out_filename = f"{input_filename.replace('.json', '')}__{joined_label}.json"
    out_path = OUTPUT_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        json.dump(rose_data, f_out, indent=2, ensure_ascii=False)

    print(f"Saved updated RoSE data with new claims => {out_path}")

    # 3.6) Save the synthetic dataset as well
    synthetic_out_filename = f"{input_filename.replace('.json', '')}__{joined_label}_synthetic.json"
    synthetic_out_path = OUTPUT_DIR / synthetic_out_filename

    with synthetic_out_path.open("w", encoding="utf-8") as f_syn:
        json.dump(synthetic_dataset, f_syn, indent=2, ensure_ascii=False)

    print(f"Saved synthetic dataset => {synthetic_out_path}")


if __name__ == "__main__":
    main()
