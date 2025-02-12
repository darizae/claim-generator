import os
import subprocess
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def test_claim_generator_cli():
    """
    Test the CLI by creating a small input file, invoking the CLI,
    and verifying the output JSON file is produced.

    Toggle the relevant lines under 'args' to test different model types:
    - HUGGINGFACE
    - OPENAI
    - JAN_LOCAL
    """

    root_dir = Path(__file__).resolve().parent
    input_dir = root_dir / "data" / "inputs"
    output_dir = root_dir / "data" / "outputs"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / "test_input.json"
    output_file = output_dir / "test_output.json"

    # Create a small input JSON file with {id, text}
    sample_input = [
        {"id": 1, "text": "Juan Arango escaped punishment from the referee for biting Jesus Zavela .\nHe could face a "
                          "retrospective punishment for the incident .\nArango had earlier scored a free kick in his "
                          "team's 4-3 defeat ."},
        {"id": 2, "text": "The spread, which was shot by photographer Mario Testino, also stars Gigi Hadid, "
                          "Ansel Elgort, Dylan Penn and her younger brother Hopper .\nThis is 19-year-old Kendall's "
                          "fifth time appearing in the pages of Vogue ."}
    ]
    with input_file.open("w", encoding="utf-8") as f:
        json.dump(sample_input, f, indent=2)

    # ----------------------------------------------------------------------------
    # ESSENTIAL CLI ARG EXAMPLES (uncomment the block you want to test):
    # ----------------------------------------------------------------------------

    args_hf_seq2seq = [
        sys.executable,  # e.g. python
        "-m", "claim_generator.cli",
        "--model-type", "huggingface",
        "--model-name-or-path", "Babelscape/t5-base-summarization-claim-extractor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    args_openai = [
        sys.executable,
        "-m", "claim_generator.cli",
        "--model-type", "openai",
        "--model-name-or-path", "gpt-3.5-turbo",
        "--api-key", openai_api_key,
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    args_jan = [
        sys.executable,
        "-m", "claim_generator.cli",
        "--model-type", "jan_local",
        "--model-name-or-path", "llama3.2-1b-instruct",
        "--endpoint-url", "http://localhost:1337/v1/chat/completions",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    # Invoke the CLI
    completed_proc = subprocess.run(
        args_jan,
        capture_output=True,
        text=True
    )

    if completed_proc.returncode != 0:
        print("CLI error output:", completed_proc.stderr)
        raise RuntimeError(f"CLI process failed with code {completed_proc.returncode}")

    print("CLI standard output:", completed_proc.stdout)

    # Check if output file exists
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found at {output_file}")

    # Parse and display output file contents
    with output_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        print("Output JSON content:", json.dumps(data, indent=2))


if __name__ == "__main__":
    test_claim_generator_cli()
