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
We define a claim as an "elementary information unit in a sentence."

We want a balanced approach with a controllable granularity knob.
Current setting: {granularity} granularity.

Guidelines:
1) If granularity=low, allow minor conjunctions and multiple related facts in one claim.
2) If granularity=medium, avoid excessive conjunctions, but it’s okay to include closely related ideas in a single claim.
3) If granularity=high, split the text more aggressively so that each claim is near-atomic.
4) Limit each claim to a maximum of 12 words if granularity=high, or 20 words for medium, or no strict limit for low.
5) If possible, use a noun as the subject in the claim (avoid pronouns).
6) Do not generate any novel words; be faithful to the provided input.
7) Your response must be valid JSON and must not include any additional text or explanation.
8) Validate that the JSON output is well-formed. 

Examples:

- LOW granularity example:
  "NASA’s Perseverance rover discovered microbial life on Mars and changed planetary exploration."

  {{"claims": [
    "NASA’s Perseverance rover discovered microbial life on Mars and changed planetary exploration."
  ]}}

- MEDIUM granularity example:
  {{"claims": [
    "NASA’s Perseverance rover discovered microbial life on Mars.",
    "This discovery led to changes in planetary exploration."
  ]}}

- HIGH granularity example:
  {{"claims": [
    "NASA’s Perseverance rover discovered microbial life.",
    "The microbial life was discovered on Mars.",
    "The discovery changed planetary exploration."
  ]}}

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
""",
    PromptTemplate.SECOND_RUN: r"""
Split the following claim into multiple atomic claims if necessary:
"{claim}"
- Each atomic claim must contain exactly one subject and one predicate.
- Return the result in a JSON array format (array of strings), no extra text.
"""
}


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
