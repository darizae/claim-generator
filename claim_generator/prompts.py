from enum import Enum


class PromptTemplate(Enum):
    DEFAULT = "default"
    MAXIMIZE_ATOMICITY = "maximize_atomicity"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    GRANULARITY = "granularity"
    SECOND_RUN = "second_run"


PROMPT_TEMPLATES = {
    PromptTemplate.DEFAULT: r""" 
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split."

For example, given the following sentence:
INPUT:
"NASA’s Perseverance rover has discovered ancient microbial life on Mars 
according to a recent study published in the journal Science. 
It established a set of new paradigms for space exploration"

OUTPUT:
{{"claims": [
  "NASA’s Perseverance rover discovered ancient microbial life.",
  "Ancient microbial life was discovered on Mars.",
  "The discovery was made according to a recent study.",
  "The study was published in the journal Science.",
  "The study established a set of new paradigms for space exploration."
]}}

Recommendations:
1) If possible, use a noun as the subject in the claim (avoid pronouns).
2) Do not generate any novel words; be faithful to the provided input.
3) Your response must be valid JSON and must not include any additional text or explanation.
4) Each fact expressed in the source text must appear as a separate claim in the output.
5) Do not include any additional text, explanations, or formatting outside the JSON object.
6) Validate that the JSON output is well-formed.

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
""",
    PromptTemplate.MAXIMIZE_ATOMICITY: r""" 
We define a claim as an "elementary information unit in a sentence, 
which no longer needs to be further split."

For Atomicity Maximization:
1) Each claim MUST contain exactly one subject and one predicate.
2) If possible, use a noun as the subject in the claim (avoid pronouns).
3) Avoid compound sentences or conjunctions (e.g., 'and', 'but', 'or').
4) If a single sentence in the source text mentions multiple facts, split them into separate claims.
5) Do not generate any novel words; be faithful to the provided input.
6) Your response must be valid JSON and must not include any additional text or explanation.
7) Validate that the JSON output is well-formed. 

Example of an extremely atomic breakdown:
INPUT:
"NASA’s Perseverance rover discovered ancient microbial life on Mars 
according to a recent study published in the journal Science. 
It established new paradigms for space exploration."

OUTPUT:
{{"claims": [
  "NASA’s Perseverance rover discovered ancient microbial life.",
  "The life discovered was microbial.",
  "The discovery happened on Mars.",
  "A recent study reported this discovery.",
  "The study was published in the journal Science.",
  "The study established new paradigms for space exploration."
]}}

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
""",
    PromptTemplate.MAXIMIZE_COVERAGE: r"""
    We define a claim as an "elementary information unit in a sentence, 
but now we want to ensure maximum coverage of all possible facts.

For Coverage Maximization:
1) Generate claims for every explicit statement in the text.
2) If a statement can be phrased in multiple ways, include multiple variants.
3) Consider logical implications as separate claims if they are suggested by the text.
4) Some redundancy is acceptable. Overlapping or restating the same fact in different words is encouraged.
5) Do not generate any novel words; be faithful to the provided input.
6) Your response must be valid JSON and must not include any additional text or explanation.
7) Validate that the JSON output is well-formed. 

Example:
INPUT:
"NASA’s Perseverance rover discovered microbial life on Mars. 
It changed how we approach planetary exploration."

OUTPUT:
{{"claims": [
  "NASA’s Perseverance rover discovered microbial life on Mars.",
  "Microbial life was discovered on Mars by Perseverance.",
  "The discovery was made by NASA’s Perseverance rover.",
  "Planetary exploration approaches changed as a result of the discovery.",
  "The discovery altered our approach to planetary exploration."
]}}

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
    """,
    PromptTemplate.GRANULARITY: r"""
We define a claim as an "elementary piece of information expressed in a sentence."

We want a balanced approach with a controllable granularity knob.
Current setting: {granularity} granularity.

Guidelines:

1) If granularity is low:
   - Combine multiple related facts into as few claims as possible.
   - It is acceptable to use conjunctions or semicolons to link several ideas in one claim.
   - No strict word limit per claim, but keep the text cohesive.
   - Example:
     "NASA’s Perseverance rover discovered microbial life on Mars, changed the approach to planetary exploration, 
      and prompted increased funding for future missions."

2) If granularity is medium:
   - Separate distinct or loosely related facts into individual claims, but you may group very closely related details together.
   - Each claim should generally not exceed about 20 words.
   - You might end up with 2–5 claims depending on the complexity of the text.
   - Example:
     "NASA’s Perseverance rover discovered microbial life on Mars.",
     "This breakthrough shifted approaches to planetary exploration and funding."

3) If granularity is high:
   - Aggressively split the text so that each claim represents only one single fact or action.
   - Limit each claim to no more than 12 words.
   - Use multiple short claims to cover all relevant details in the source text.
   - Example:
     "NASA’s Perseverance rover discovered microbial life.",
     "The discovery occurred on Mars.",
     "It reshaped planetary exploration.",
     "Future missions received more funding."

Additional Instructions:
- Always use a noun as the subject in each claim when possible (avoid pronouns).
- Be faithful to the provided input text (do not introduce new words).
- Your response must be valid JSON and include no additional text or explanation.
- Ensure that the JSON output is well-formed and contains an array called "claims".

More Detailed Examples:

- **LOW granularity example**:
  ```json
  {{
    "claims": [
      "NASA’s Perseverance rover discovered ancient microorganisms, changed how we explore planets, and influenced future funding decisions for Mars missions."
    ]
  }}
MEDIUM granularity example:
{{
  "claims": [
    "NASA’s Perseverance rover discovered ancient microorganisms on Mars.",
    "That finding led to shifts in exploration strategies and funding."
  ]
}}
HIGH granularity example:
{{
  "claims": [
    "NASA’s Perseverance rover discovered ancient microorganisms.",
    "They were found on Mars.",
    "The discovery altered exploration strategies.",
    "Funding priorities were also impacted."
  ]
}}
Now do the same for this input:

INPUT: {SOURCE_TEXT}

OUTPUT:
"""
}

TRIPLE_TO_CLAIM_PROMPT = r"""
("system",
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


def get_prompt_template(template: PromptTemplate, **kwargs) -> str:
    """
    Returns the formatted prompt template corresponding to the given enum value.

    Additional formatting parameters (e.g., SOURCE_TEXT, granularity, claim) can be provided as keyword arguments.

    Raises:
        ValueError: If the template enum is not recognized.
    """
    try:
        template_str = PROMPT_TEMPLATES[template]
    except KeyError:
        raise ValueError(f"Unrecognized prompt template: {template}")

    return template_str.format(**kwargs)
