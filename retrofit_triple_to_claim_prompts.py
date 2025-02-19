#!/usr/bin/env python3
"""
retrofit_triple_to_claim_prompts.py

This script fills in the 'triple_to_claim_prompts' field for entries in
'rose_datasets_kg_claims_synthetic.json' that previously had
"unavailable (already generated; no stored triple data)".

It does NOT call the LLM; it simply builds the prompt strings.

Steps:
1) Load 'rose_datasets_kg_claims_synthetic.json'.
2) For each entry where 'triple_to_claim_prompts' == "unavailable (already generated; no stored triple data)",
   parse each triple from kg_output["triples"] (dictionary-based: {subject, predicate, object}),
   and embed it in the original TRIPLE_TO_CLAIM_PROMPT used by claim-generator:
        ("system", ...)
        ("user", "Triple: [{subject}, {predicate}, {object}]", ...)
3) Store those per-triple prompt strings in 'entry["triple_to_claim_prompts"]' as a list.
4) Partial-save every 20 updated entries.

Usage:
    python retrofit_triple_to_claim_prompts.py

Requirements:
    - The file 'rose_datasets_kg_claims_synthetic.json' in data/outputs.
"""

import os
import json
from pathlib import Path

###############################################################################
# 1) Configuration
###############################################################################
DATA_DIR = Path("data") / "outputs"
INPUT_FILE = "rose_datasets_kg_claims_synthetic.json"
SAVE_EVERY = 20  # Save partial progress every 20 updated entries
PRINT_PROGRESS = True

###############################################################################
# 2) The original triple->claim prompt from claim_generator
###############################################################################
TRIPLE_TO_CLAIM_PROMPT = r"""("system",
""
Convert a triple into a short, standalone factual statement. 
Return only the statement text. 
Do not add JSON or extraneous formatting.
""
),
("user",
""
Triple: [{subject}, {predicate}, {object}]
""
),
"""

###############################################################################
# 3) Main Script
###############################################################################
def main():
    # Load the synthetic dataset
    synthetic_path = DATA_DIR / INPUT_FILE
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Could not find file: {synthetic_path}")

    with synthetic_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    updated_count = 0

    for entry in data:
        # We only proceed if triple_to_claim_prompts is the "unavailable" placeholder
        t2c_prompts = entry.get("triple_to_claim_prompts", "")
        if not isinstance(t2c_prompts, str):
            # Already a list or some other structure => skip
            continue
        if not t2c_prompts.startswith("unavailable"):
            # Some other string => skip
            continue

        kg_output = entry.get("kg_output", {})
        if not isinstance(kg_output, dict):
            # If kg_output is missing or not a dict, skip
            continue

        triples = kg_output.get("triples", [])
        if not isinstance(triples, list) or len(triples) == 0:
            # No valid triple array => skip
            continue

        # Build the array of prompt strings
        triple_prompt_list = []
        for triple in triples:
            subj = triple.get("subject", "").strip()
            pred = triple.get("predicate", "").strip()
            obj = triple.get("object", "").strip()

            # Insert into the original triple->claim prompt
            prompt_str = TRIPLE_TO_CLAIM_PROMPT.format(subject=subj, predicate=pred, object=obj)
            triple_prompt_list.append(prompt_str)

        # Store it
        entry["triple_to_claim_prompts"] = triple_prompt_list

        updated_count += 1
        # Partial save
        if updated_count % SAVE_EVERY == 0:
            if PRINT_PROGRESS:
                print(f"[INFO] Updated {updated_count} entries out of {total_entries}; saving partial results.")
            with synthetic_path.open("w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

    # Done
    print(f"[INFO] Processed {total_entries} entries; updated {updated_count}.")
    print(f"[INFO] Saving final data to {synthetic_path}")
    with synthetic_path.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)
    print("[INFO] Complete!")


if __name__ == "__main__":
    main()
