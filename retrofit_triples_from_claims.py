#!/usr/bin/env python3
"""
retrofit_triples_from_claims.py

1) Reads 'rose_datasets_kg_claims_synthetic.json'.
2) For each entry with "kg_output" == "unavailable (...)",
   we parse each claim to produce exactly one triple in the format:
     {
       "subject": "....",
       "predicate": "....",
       "object": "...."
     }
   Then "kg_output" becomes:
     {
       "triples": [ { ... }, { ... }, ... ]
     }

3) We also set "kg_parser_prompt" using the original REFINED_CLAIM_PROMPT,
   as if the text had been passed to your kg-parser.

4) We do NOT store any triple retrofit prompts. We just do quick calls
   to turn each claim into a triple.

5) Saves partial progress every 20 entries.

Usage:
    python retrofit_triples_from_claims.py
"""

import os
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()
from dotenv import load_dotenv

# We'll import the original KG parser prompt
# so we can store it under "kg_parser_prompt".
from kg_parser.prompt_templates import REFINED_CLAIM_PROMPT

###############################################################################
# 1) Configuration
###############################################################################
INPUT_FILE = "rose_datasets_kg_claims_synthetic.json"
DATA_DIR = Path("data") / "outputs"
SAVE_EVERY = 20  # partial save after every 20 updates
PRINT_PROGRESS = True

# Load environment for OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in environment. Please set it or supply it.")


###############################################################################
# 2) Helper Function: convert a single claim -> {subject, predicate, object}
###############################################################################
# This is a minimal prompt that tries to parse exactly one claim into a triple.


def claim_to_triple_dict(claim, model_name="gpt-3.5-turbo", temperature=0.0):
    """
    Convert one short claim into a triple dictionary:
       { "subject": "...", "predicate": "...", "object": "..." }

    We do a quick ChatCompletion call. If parsing fails, we return
    a fallback triple dict or empty fields.
    """
    system_prompt = """You are an expert at converting a short claim into a knowledge graph triple. 
Return valid JSON with keys "subject", "predicate", "object" only. 
No additional text, no commentary."""
    user_prompt = f"""Claim: "{claim}"

Output a single JSON with the keys: "subject", "predicate", "object". 
Example:
{{
  "subject": "Amanda Jackson",
  "predicate": "was born on",
  "object": "June 1, 1985"
}}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(model=model_name,
                                                  messages=messages,
                                                  temperature=temperature)
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        # Basic checks
        if not all(k in data for k in ("subject", "predicate", "object")):
            return {
                "subject": "",
                "predicate": "",
                "object": ""
            }
        return data
    except Exception as e:
        if PRINT_PROGRESS:
            print(f"[ERROR] Could not parse claim => triple. Claim = '{claim}'. Error = {e}")
        # fallback
        return {
            "subject": "",
            "predicate": "",
            "object": ""
        }


###############################################################################
# 3) Main Script
###############################################################################
def main():
    # Load the synthetic dataset
    synthetic_path = DATA_DIR / INPUT_FILE
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Could not find synthetic file: {synthetic_path}")

    with synthetic_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    total_entries = len(data)
    for idx, entry in enumerate(data, start=1):
        kg_output = entry.get("kg_output", "")
        if not isinstance(kg_output, str):
            # Already a dictionary or actual triple data => skip
            continue
        if not kg_output.startswith("unavailable"):
            # This entry presumably has real data => skip
            continue

        claims = entry.get("claims", [])
        if not claims:
            # Nothing to reconstruct
            continue

        # (A) Build the new "kg_output" => dict with "triples" -> list of triple dicts
        triple_dicts = []
        no_of_claims = len(claims)
        for claim in claims:
            triple_res = claim_to_triple_dict(claim)
            triple_dicts.append(triple_res)

        # (B) Assign them to the entry:
        entry["kg_output"] = {
            "triples": triple_dicts
        }

        # (C) Also set "kg_parser_prompt" using your original template
        #     (like it was used for the reference).
        reference_text = entry.get("reference", "")
        parser_prompt = REFINED_CLAIM_PROMPT.format(input=reference_text)
        entry["kg_parser_prompt"] = parser_prompt

        updated_count += 1
        if PRINT_PROGRESS:
            print(f"[INFO] Reconstructed {len(triple_dicts)} triple(s) for record_id={entry.get('record_id')}")

        # (D) Partial save
        if updated_count % SAVE_EVERY == 0:
            print(f"[INFO] Processed {updated_count} updated entries so far. Saving partial results...")
            with synthetic_path.open("w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

    # 4) Final save
    print(f"[INFO] Finished scanning {total_entries} items. Updated {updated_count} entries.")
    print(f"[INFO] Saving final results to {synthetic_path}")
    with synthetic_path.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)
    print("[INFO] Complete!")


if __name__ == "__main__":
    main()
