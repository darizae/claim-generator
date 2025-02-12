from enum import Enum


class PromptTemplate(Enum):
    DEFAULT = "default"
    # Additional placeholders for future prompts
    # E.g. EXTENDED = "extended"
    # Or any other named prompt


def get_prompt_template(template: PromptTemplate) -> str:
    """
    Return the text of the prompt template corresponding to the enum value.
    """
    if template == PromptTemplate.DEFAULT:
        # The refined claim prompt from your original code
        return """ 
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
8 )Validate that the JSON output is well-formed.

Now do the same for this input:

INPUT:
{SOURCE_TEXT}

OUTPUT:
"""
    # If you add more prompt templates, handle them below.
    raise ValueError(f"Unrecognized prompt template: {template}")
