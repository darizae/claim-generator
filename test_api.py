import os

from claim_generator import ModelConfig, ModelType, create_generator, PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_config = ModelConfig(
    model_type=ModelType.OPENAI,
    model_name_or_path="gpt-3.5-turbo",
    api_key=openai_api_key
)

jan_local_config = ModelConfig(
    model_type=ModelType.JAN_LOCAL,
    model_name_or_path="llama3.2-1b-instruct",
    endpoint_url="http://localhost:1337/v1/chat/completions",
    temperature=0.7,
    max_length=512,
    batch_size=8
)

hf_seq2seq_config = ModelConfig(
    model_type=ModelType.HUGGINGFACE,
    model_name_or_path="Babelscape/t5-base-summarization-claim-extractor",
    max_length=512,
    batch_size=8
)


generator = create_generator(openai_config, PromptTemplate.DEFAULT)

texts = [
    "Juan Arango escaped punishment from the referee for biting Jesus Zavela .\nHe could face a retrospective "
    "punishment for the incident .\nArango had earlier scored a free kick in his team's 4-3 defeat .",
    "The spread, which was shot by photographer Mario Testino, also stars Gigi Hadid, Ansel Elgort, Dylan Penn and "
    "her younger brother Hopper .\nThis is 19-year-old Kendall's fifth time appearing in the pages of Vogue ."
]
results = generator.generate_claims(texts)
print(results)
