#!/usr/bin/env python3
"""
retrofit_triples_from_claims.py

This script:
1) Reads 'rose_datasets_kg_claims_synthetic.json'.
2) Finds entries where 'kg_output' == "unavailable (already generated; no stored triple data)".
3) For each such entry, we pass the claims to OpenAI, asking it to produce an
   array of dictionary-based triples, each with keys 'subject', 'predicate', 'object'.
4) We store the result under entry["kg_output"] as:
   {
     "triples": [
       { "subject": ..., "predicate": ..., "object": ...},
       ...
     ]
   }
5) We do NOT store or update 'triple_to_claim_prompts' in this script.
   We also do NOT store a prompt for the retrofit call itself,
   because you said you do not need that in the final JSON.
6) We save partial progress every 20 entries.

Usage:
    python retrofit_triples_from_claims.py

Requirements:
    - OPENAI_API_KEY in environment or .env file
    - The file 'rose_datasets_kg_claims_synthetic.json' in data/outputs
"""

import os
import json
from pathlib import Path
from openai import OpenAI

from claim_generator.prompts import TRIPLE_TO_CLAIM_PROMPT

client = OpenAI()
from dotenv import load_dotenv

###############################################################################
# 1) Configuration
###############################################################################
INPUT_FILE = "rose_datasets_kg_claims_synthetic.json"
DATA_DIR = Path("data") / "outputs"

SAVE_EVERY = 20  # Save partial progress after every 20 updated entries

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found. Please set it in environment or .env.")

###############################################################################
# 2) Prompt Template for Reconstructing Dictionary-Based Triples
###############################################################################
SYSTEM_PROMPT = """You are an expert at converting short factual statements into knowledge graph triples in dictionary form. 
Each triple must have exactly three keys: "subject", "predicate", and "object". 
Do not add any extra keys. Return valid JSON with a single 'triples' array of these dictionaries.
"""

USER_PROMPT_TEMPLATE = """
You will receive a list of claims (each is a short factual statement). For each claim, produce exactly one triple in this structure:
{{ "subject": "S", "predicate": "P", "object": "O" }}

Return a single JSON object with one key "triples", whose value is an array of these dictionaries. 
Include no additional commentary or fields.
If no object is explicitly given, set the object to ‘[No explicit object]’ or ‘None’.
Never leave the object blank.

Example:
Claims:
1) "Amanda Jackson was born on June 1, 1985."
2) "Amanda Jackson is a basketball player."

Output:
{{
  "triples": [
    {{ "subject": "Amanda Jackson", "predicate": "born on", "object": "June 1, 1985" }},
    {{ "subject": "Amanda Jackson", "predicate": "occupation", "object": "basketball player" }}
  ]
}}

Now process these claims:
{claims_str}
Make sure the JSON is valid, has exactly one key 'triples', and each triple is a dictionary 
with 'subject', 'predicate', 'object'.
"""


def reconstruct_dict_triples_for_claims(claims, model_name="gpt-3.5-turbo", temperature=0.0):
    """
    Calls OpenAI once for the entire list of claims, returning a dict like:
       {
         "triples": [
           {
             "subject": "...",
             "predicate": "...",
             "object": "..."
           },
           ...
         ]
       }
    If it fails or doesn't produce the expected keys, returns a fallback with an empty array.
    """
    enumerated_claims = []
    for i, c in enumerate(claims, start=1):
        enumerated_claims.append(f"{i}) \"{c}\"")
    claims_str = "\n".join(enumerated_claims)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(claims_str=claims_str)},
    ]

    try:
        response = client.chat.completions.create(model=model_name,
                                                  messages=messages,
                                                  temperature=temperature)
        content = response.choices[0].message.content.strip()
        if "None" in content:
            content = content.replace(": None", ": null")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Provide more context
            raise ValueError(
                f"[ERROR] JSON decoding failed. Possibly invalid JSON. "
                f"Raw response:\n{content}\n\nError: {e}"
            ) from e

        # 2) Must have "triples" as a list
        if "triples" not in data or not isinstance(data["triples"], list):
            return {"triples": []}

        validated_triples = []
        for idx, t in enumerate(data["triples"]):
            if not isinstance(t, dict):
                continue

            subj = t.get("subject", "")
            pred = t.get("predicate", "")
            obj = t.get("object", "")

            # 3) Fallback for None or other non-string => placeholders
            if not isinstance(subj, str):
                subj = "[No explicit subject]"
            if not isinstance(pred, str):
                pred = "[No explicit predicate]"
            if not isinstance(obj, str):
                obj = "[No explicit object]"

            # 4) Safe strip
            subj = subj.strip()
            pred = pred.strip()
            obj = obj.strip()

            # 5) Rebuild triple
            triple_dict = {
                "subject": subj,
                "predicate": pred,
                "object": obj
            }
            validated_triples.append(triple_dict)

        return {"triples": validated_triples}

    except Exception as e:
        raise ValueError(
            f"[ERROR] Reconstructing dictionary-based triples failed.\n"
            f"Details: {e}"
        ) from e


###############################################################################
# 3) Main Script
###############################################################################
def main():
    synthetic_path = DATA_DIR / INPUT_FILE
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Could not find synthetic file: {synthetic_path}")

    with synthetic_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    total_count = 0
    for entry in data:
        total_count += 1
        kg_output = entry.get("kg_output", "")
        if not isinstance(kg_output, str):
            # Already has dictionary-based triple data => skip
            continue
        if not kg_output.startswith("unavailable"):
            # Some other string or already replaced => skip
            continue

        claims = entry.get("claims", [])
        if not claims:
            # No claims => can't reconstruct anything
            continue

        # 1) Call OpenAI to get dictionary-based triples
        ret_data = reconstruct_dict_triples_for_claims(claims)
        # 2) Store them in "kg_output"
        entry["kg_output"] = ret_data  # e.g. {"triples": [{ subj, pred, obj }, ...]}

        # (B) Build triple_to_claim_prompts: for each triple, embed it in TRIPLE_TO_CLAIM_PROMPT
        triple_prompts = []
        for t in ret_data["triples"]:
            subj = t.get("subject", "").strip()
            pred = t.get("predicate", "").strip()
            obj = t.get("object", "").strip()
            triple_prompt = TRIPLE_TO_CLAIM_PROMPT.format(
                subject=subj,
                predicate=pred,
                object=obj
            )
            triple_prompts.append(triple_prompt)

        entry["triple_to_claim_prompts"] = triple_prompts

        updated_count += 1
        print(f"[INFO] Reconstructed {len(ret_data['triples'])} triple(s) for record_id={entry.get('record_id')}")
        if updated_count % SAVE_EVERY == 0:
            print(f"[INFO] Updated {updated_count} entries so far; saving partial results.")
            with synthetic_path.open("w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

    print(f"[INFO] Done scanning {total_count} entries. Updated {updated_count}.")
    print(f"[INFO] Saving final results to {synthetic_path}")
    with synthetic_path.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)
    print("[INFO] Complete!")


if __name__ == "__main__":
    main()
