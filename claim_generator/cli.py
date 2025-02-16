import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from .config import ModelConfig, ModelType, PathConfig
from .prompts import PromptTemplate
from .generator import create_generator


def main():
    parser = argparse.ArgumentParser(description="CLI for claim generation.")
    parser.add_argument("--model-type", type=str, default="huggingface",
                        choices=[m.value for m in ModelType],
                        help="Which model type to use: huggingface, openai, jan_local.")
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="HF model name/path or OpenAI model (e.g. 'gpt-3.5-turbo').")
    parser.add_argument("--prompt-template", type=str, default="default",
                        help="Which prompt template to use, e.g. 'default' (for now).")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (for OpenAI or Jan if needed).")
    parser.add_argument("--endpoint-url", type=str, default=None,
                        help="Endpoint url for Jan local usage.")
    parser.add_argument("--device", type=str, default=None,
                        help="Requested device: 'cuda', 'mps', 'cpu', etc. (auto if None)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max input length for truncation.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to JSON file with an array of {id, text}.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Where to save the output JSON.")
    args = parser.parse_args()

    # 1) Parse the model type
    try:
        model_type = ModelType(args.model_type)
    except ValueError:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # 2) Parse the prompt template
    #  (We only have 'default' for now, but you can easily extend.)
    prompt_template = PromptTemplate.DEFAULT
    # If you'd eventually have other templates, you'd do:
    # prompt_template = PromptTemplate(args.prompt_template.upper())
    # or some mapping from string => enum.

    # 3) Construct ModelConfig
    config = ModelConfig(
        model_type=model_type,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        api_key=args.api_key,
        endpoint_url=args.endpoint_url,
        temperature=args.temperature,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    # 4) Create generator
    generator = create_generator(config, prompt_template)

    # 5) Load input data
    input_path = Path(args.input_file)
    with input_path.open("r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Expecting a list of objects with {id, text}
    texts = [item["text"] for item in input_data]

    # 6) Generate
    claims_list = generator.generate_claims(texts)

    # 7) Build output data
    output_data: List[Dict[str, Any]] = []
    for item, claims in zip(input_data, claims_list):
        output_data.append({
            "id": item["id"],
            "text": item["text"],
            "claims": claims,
            "model_config": {
                "model_type": config.model_type.value,
                "model_name_or_path": config.model_name_or_path,
                "device": config.device,
                "temperature": config.temperature,
                "max_length": config.max_length,
                "batch_size": config.batch_size
            }
        })

    # 8) Write output file
    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Wrote {len(output_data)} items to {output_path}.")


if __name__ == "__main__":
    main()
