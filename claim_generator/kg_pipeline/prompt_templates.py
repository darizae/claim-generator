REFINED_CLAIM_PROMPT = r"""
("system",
""
You are an expert at extracting information in structured formats to build a knowledge graph.
Step 1 – Entity detection: Identify all entities in the raw text. Entities should be basic and unambiguous, similar to Wikipedia nodes.
Step 2 – Coreference resolution: Determine which expressions in the text refer to the same entity. Avoid duplicates by selecting only the most specific form.
Step 3 – Relation extraction: Identify semantic relationships between the entities.
Format: Return the knowledge graph as a JSON object with a single key "triples". The value must be a list of triples, where each triple is a list of exactly three non-empty strings, for example: ["entity 1", "relation", "entity 2"].
The output must be valid JSON and include nothing but the JSON object.
""
,
"human",
"Use the above instructions to extract a knowledge graph from the following input. Return only the JSON object without any extra text or commentary."
,
"human",
""
Important Tips:
- Ensure that every relevant piece of information is captured.
- Each triple must contain exactly three non-empty strings.
- Do not include any additional text, explanations, or formatting outside the JSON object.
- Validate that the JSON output is well-formed.
""
),
("human",
""
Here are some example input and output pairs.
Example 1.
Input:
"The Walt Disney Company, commonly known as Disney, is an American multinational mass media and entertainment conglomerate that is headquartered at the Walt Disney Studios complex in Burbank, California."
Output:
{{
  "triples": [
    ["The Walt Disney Company", "headquartered at", "Walt Disney Studios complex in Burbank, California"],
    ["The Walt Disney Company", "commonly known as", "Disney"],
    ["The Walt Disney Company", "instance of", "American multinational mass media and entertainment conglomerate"]
  ]
}}

Example 2.
Input:
"Amanda Jackson was born in Springfield, Ohio, USA on June 1, 1985. She was a basketball player for the U.S. women’s team."
Output:
{{
  "triples": [
    ["Amanda Jackson", "born in", "Springfield, Ohio, USA"],
    ["Amanda Jackson", "born on", "June 1, 1985"],
    ["Amanda Jackson", "occupation", "basketball player"],
    ["Amanda Jackson", "played for", "U.S. women’s basketball team"]
  ]
}}

Example 3.
Input:
"Music executive Darius Van Arman was born in Pennsylvania. He attended Gonzaga College High School and is a human being."
Output:
{{
  "triples": [
    ["Darius Van Arman", "occupation", "Music executive"],
    ["Darius Van Arman", "born in", "Pennsylvania"],
    ["Darius Van Arman", "attended", "Gonzaga College High School"],
    ["Darius Van Arman", "instance of", "human being"]
  ]
}}


Example 4.
Input:
"Italy had 3.6x times more cases of coronavirus than China."
Output:
{{
  "triples": [
    ["Italy", "had 3.6x times more cases of coronavirus than", "China"]
  ]
}}

Now, process the following input:
Input: <input>{input}</input>
""
)
"""
