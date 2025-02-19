#!/usr/bin/env python3
"""
remove_click_here.py

Removes any claim/string containing the substrings "CLICK HERE" or "Click here"
(case-sensitive) from the RoSE dataset (except from the 'reference' key).

Steps:
  1) Load rose_datasets_with_claims_with_click.json
  2) Skip or ignore the 'reference' key for removal
  3) If a field's value is a single string containing "CLICK HERE" or "Click here",
     remove that key from the entry entirely
  4) If a field's value is a list of strings, remove only the items containing those substrings
  5) Log the removals (record_id, key, and the removed string)
  6) Save the cleaned result to rose_datasets_with_claims.json
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='remove_click_here.log',
    filemode='w',
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = "rose_datasets_with_claims_with_click.json"
OUTPUT_FILE = "rose_datasets_with_claims.json"

TARGET_SUBSTRINGS = ["CLICK HERE", "Click here"]  # case-sensitive checks


def contains_click_here(text: str) -> bool:
    """
    Returns True if the given text contains either
    "CLICK HERE" or "Click here" as a substring (case-sensitive).
    """
    for substr in TARGET_SUBSTRINGS:
        if substr in text:
            return True
    return False


def clean_entry(entry: dict, record_id: str) -> dict:
    """
    Removes any strings (across the entry's keys) containing
    "CLICK HERE" or "Click here", except for the 'reference' key.
    - If a key has a single string value containing the substrings, remove that key entirely.
    - If a key has a list of strings, remove the elements containing those substrings.
    Returns the modified entry.
    """
    keys_to_remove = []

    # We'll build up changes in a new dict so we don't mutate while iterating
    cleaned_entry = {}

    for key, value in entry.items():
        # Skip the 'reference' key
        if key == "reference" or key == "source":
            # Always keep 'reference' unmodified
            cleaned_entry[key] = value
            continue

        # If it's a single string, check and possibly remove the entire key
        if isinstance(value, str):
            if contains_click_here(value):
                logger.info(f"Removing key='{key}' in record_id='{record_id}' because it contains '{value}'")
                keys_to_remove.append(key)
            else:
                cleaned_entry[key] = value

        # If it's a list of strings, remove just the items that contain "CLICK HERE"/"Click here"
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            cleaned_list = []
            for item in value:
                if contains_click_here(item):
                    logger.info(f"Removing string from key='{key}' in record_id='{record_id}': '{item}'")
                else:
                    cleaned_list.append(item)

            cleaned_entry[key] = cleaned_list

        else:
            # For other data types (e.g., lists of objects, dicts, etc.),
            # you can decide what to do. By default we just keep them unmodified.
            cleaned_entry[key] = value

    return cleaned_entry


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        logger.error(f"Input file '{input_path}' not found.")
        return

    logger.info(f"Loading dataset from '{input_path}'...")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data is generally { "dataset_name": [entries...], "dataset_name2": [...], ... }
    cleaned_data = {}

    for dataset_name, entries in data.items():
        if not isinstance(entries, list):
            logger.warning(f"'{dataset_name}' is not a list; skipping.")
            continue

        cleaned_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning(f"Found non-dict entry in dataset '{dataset_name}'; skipping.")
                continue

            record_id = entry.get("record_id", "NO_RECORD_ID")
            cleaned_entries.append(clean_entry(entry, record_id))

        cleaned_data[dataset_name] = cleaned_entries

    logger.info(f"Saving cleaned dataset to '{output_path}'...")
    with output_path.open("w", encoding="utf-8") as f_out:
        json.dump(cleaned_data, f_out, indent=2, ensure_ascii=False)

    logger.info("Done. Please check remove_click_here.log for details of what was removed.")


if __name__ == "__main__":
    main()
