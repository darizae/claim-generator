#!/usr/bin/env python3
"""
validate_datasets.py

Validates the structure and data in:
1) rose_datasets_with_claims.json
2) rose_datasets_kg_claims_synthetic.json

Checks performed:
  - No duplicate record_id
  - No empty values for any key
  - Validates data types for each key
  - Reorders keys in the synthetic dataset to:
       record_id -> dataset_name -> reference -> kg_parser_prompt
       -> kg_output -> triple_to_claim_prompts -> claims
  - Logs all issues to validation.log

Written as an example. Adjust the exact checks as needed.
"""

import json
import logging
from collections import defaultdict

# ----------------------------------------------------------------------
# Setup logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    filename='validation.log',
    filemode='w',  # overwrite on each run, or use 'a' to append
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_rose_datasets_with_claims(path: str):
    """
    Validates the structure of rose_datasets_with_claims.json.
    Returns the loaded JSON (dict) if valid (or partially valid).
    Logs issues to validation.log.
    """
    logger.info(f"Validating {path} ...")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    record_ids_seen = set()
    total_issues = 0

    # Expected keys for an entry (some might be optional in your real data;
    # modify as needed).
    # This dict maps key -> (type or "list_of_strings"), whether it's optional, etc.
    expected_structure = {
        "source": {"type": str, "optional": False},
        "reference": {"type": str, "optional": False},
        "record_id": {"type": str, "optional": False},
        "reference_acus_deduped_0.7_select_longest": {"type": "list_of_strings", "optional": True},
        "gpt_default": {"type": "list_of_strings", "optional": True},
        "gpt_maximize_atomicity": {"type": "list_of_strings", "optional": True},
        "gpt_maximize_coverage": {"type": "list_of_strings", "optional": True},
        "gpt_granularity_low": {"type": "list_of_strings", "optional": True},
        "gpt_granularity_high": {"type": "list_of_strings", "optional": True},
        "kg_based_claims": {"type": "list_of_strings", "optional": True}
    }

    # data is typically { "some_dataset_name": [ entries... ], ... }
    for dataset_name, entries in data.items():
        if not isinstance(entries, list):
            logger.error(f"Dataset '{dataset_name}' is not a list. Skipping.")
            total_issues += 1
            continue

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.error(f"Entry at index {idx} in dataset '{dataset_name}' is not a dict. Skipping.")
                total_issues += 1
                continue

            # --- Check for duplicates ---
            rec_id = entry.get("record_id")
            if rec_id in record_ids_seen:
                logger.error(f"Duplicate record_id '{rec_id}' found in dataset '{dataset_name}'")
                total_issues += 1
            else:
                record_ids_seen.add(rec_id)

            # --- Validate each expected key ---
            for key, spec in expected_structure.items():
                is_optional = spec["optional"]
                required_type = spec["type"]

                if key not in entry:
                    if not is_optional:
                        logger.error(f"Missing required key '{key}' in record_id='{rec_id}' (dataset '{dataset_name}')")
                        total_issues += 1
                    # If optional and not present, that's OK; skip further checks
                    continue

                val = entry[key]

                # Check empty
                if val == "" or (isinstance(val, list) and len(val) == 0):
                    logger.error(f"Key '{key}' in record_id='{rec_id}' has an empty value.")
                    total_issues += 1

                # Check type
                if required_type == "list_of_strings":
                    if not isinstance(val, list):
                        logger.error(f"Key '{key}' in record_id='{rec_id}' should be a list but got {type(val)}.")
                        total_issues += 1
                    else:
                        # ensure each element is a non-empty string
                        for elem in val:
                            if not isinstance(elem, str) or not elem.strip():
                                logger.error(f"Key '{key}' in record_id='{rec_id}' has a non-string or empty string element.")
                                total_issues += 1
                else:
                    # For example, if required_type == str
                    if not isinstance(val, required_type):
                        logger.error(f"Key '{key}' in record_id='{rec_id}' expected {required_type} but got {type(val)}.")
                        total_issues += 1

    if total_issues > 0:
        logger.warning(f"Validation for '{path}' completed with {total_issues} issue(s).")
    else:
        logger.info(f"Validation for '{path}' completed successfully with no issues.")

    return data


def validate_and_reorder_rose_datasets_kg_claims_synthetic(path: str, output_path: str):
    """
    Validates the structure of rose_datasets_kg_claims_synthetic.json (which is a list).
    - Checks no duplicate record_id
    - Ensures no empty values
    - Ensures data type correctness
    - Re-orders keys in each object
    Writes out the re-ordered JSON to `output_path`.
    """
    logger.info(f"Validating {path} ...")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error(f"Top-level data in '{path}' should be a list, found {type(data)}.")
        return

    record_ids_seen = set()
    total_issues = 0

    # We want the final order of keys to be exactly:
    ordered_keys = [
        "record_id",
        "dataset_name",
        "reference",
        "kg_parser_prompt",
        "kg_output",
        "triple_to_claim_prompts",
        "claims"
    ]

    # Let’s define the expected structure for each key:
    expected_structure = {
        "record_id": {"type": str, "optional": False},
        "dataset_name": {"type": str, "optional": False},
        "reference": {"type": str, "optional": False},
        "kg_parser_prompt": {"type": str, "optional": False},
        # kg_output is an object with 'triples' -> list of triple objects
        "kg_output": {"type": dict, "optional": False},
        # triple_to_claim_prompts is a list of strings
        "triple_to_claim_prompts": {"type": "list_of_strings", "optional": False},
        # claims is a list of strings
        "claims": {"type": "list_of_strings", "optional": False},
    }

    # We'll store the updated entries in new_data
    new_data = []

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.error(f"Entry at index {idx} is not a dict. Skipping.")
            total_issues += 1
            continue

        rec_id = entry.get("record_id", None)
        if rec_id in record_ids_seen:
            logger.error(f"Duplicate record_id '{rec_id}' found in synthetic dataset.")
            total_issues += 1
        else:
            record_ids_seen.add(rec_id)

        # Validate the presence and structure of each required key
        for key, spec in expected_structure.items():
            is_optional = spec["optional"]
            required_type = spec["type"]

            if key not in entry:
                if not is_optional:
                    logger.error(f"Missing required key '{key}' in record_id='{rec_id}'.")
                    total_issues += 1
                continue

            val = entry[key]
            # Check empty
            if val == "" or (isinstance(val, list) and len(val) == 0):
                logger.error(f"Key '{key}' in record_id='{rec_id}' is empty.")
                total_issues += 1

            # Check type
            if required_type == "list_of_strings":
                if not isinstance(val, list):
                    logger.error(f"Key '{key}' in record_id='{rec_id}' should be a list but got {type(val)}.")
                    total_issues += 1
                else:
                    for item in val:
                        if not isinstance(item, str) or not item.strip():
                            logger.error(f"Key '{key}' in record_id='{rec_id}' has a non-string or empty string item.")
                            total_issues += 1
            elif required_type == dict:
                if not isinstance(val, dict):
                    logger.error(f"Key '{key}' in record_id='{rec_id}' should be a dict but got {type(val)}.")
                    total_issues += 1
                else:
                    # If it's kg_output, we expect val["triples"] -> a list of {subject,predicate,object}
                    if key == "kg_output":
                        if "triples" not in val:
                            logger.error(f"kg_output in record_id='{rec_id}' has no 'triples' key.")
                            total_issues += 1
                        else:
                            # Check that triples is a list of dict objects with subject, predicate, object
                            triples_val = val["triples"]
                            if not isinstance(triples_val, list):
                                logger.error(f"'triples' in kg_output for record_id='{rec_id}' should be a list.")
                                total_issues += 1
                            else:
                                for t_idx, triple_obj in enumerate(triples_val):
                                    if not isinstance(triple_obj, dict):
                                        logger.error(f"Triple #{t_idx} in record_id='{rec_id}' is not a dict.")
                                        total_issues += 1
                                        continue
                                    # check for subject/predicate/object
                                    for tp_key in ["subject", "predicate", "object"]:
                                        if tp_key not in triple_obj:
                                            logger.error(
                                                f"Triple #{t_idx} in record_id='{rec_id}' missing '{tp_key}'."
                                            )
                                            total_issues += 1
                                        else:

                                            MISSING_OBJ_AS_ERROR = False

                                            if MISSING_OBJ_AS_ERROR:
                                                if not triple_obj[tp_key] or not isinstance(triple_obj[tp_key], str):
                                                    logger.error(
                                                        f"Triple #{t_idx} in record_id='{rec_id}' has empty or non-string '{tp_key}'."
                                                    )
                                                    total_issues += 1
                                            else:
                                                pass

            elif required_type == str:
                if not isinstance(val, str):
                    logger.error(f"Key '{key}' in record_id='{rec_id}' should be a string but got {type(val)}.")
                    total_issues += 1

        # Reorder keys in the entry:
        # We’ll create a new dictionary with the specified key order,
        # then append any additional keys after (if they exist).
        reordered_entry = {}
        for k in ordered_keys:
            if k in entry:
                reordered_entry[k] = entry[k]

        # If there are any other keys in the original entry not in ordered_keys,
        # you can choose to drop them or keep them. Here we keep them at the end.
        for k, v in entry.items():
            if k not in ordered_keys:
                reordered_entry[k] = v

        new_data.append(reordered_entry)

    if total_issues > 0:
        logger.warning(f"Validation for '{path}' completed with {total_issues} issue(s).")
    else:
        logger.info(f"Validation for '{path}' completed successfully with no issues.")

    # Write out the re-ordered data
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(new_data, f_out, indent=2, ensure_ascii=False)

    return new_data


def main():
    # Adjust file names/paths to match your setup
    path_rose_with_claims = "rose_datasets_with_claims_no_click.json"
    path_kg_claims_synthetic = "rose_datasets_kg_claims_synthetic.json"
    path_kg_claims_synthetic_out = "rose_datasets_kg_claims_synthetic.json"

    # 1) Validate the main "with_claims" dataset
    _ = validate_rose_datasets_with_claims(path_rose_with_claims)

    # 2) Validate & Reorder the synthetic dataset
    _ = validate_and_reorder_rose_datasets_kg_claims_synthetic(
        path_kg_claims_synthetic,
        path_kg_claims_synthetic_out
    )

    logger.info("Validation script completed.")


if __name__ == "__main__":
    main()
