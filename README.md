# Claim Generator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)](#)
[![Code Style PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

A Python library and CLI tool designed for generating **structured factual claims** from text using Large Language Models (LLMs) and Knowledge Graphs (KGs).

---

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
   - [Python API Usage](#python-api-usage)
   - [CLI Usage](#cli-usage)
4. [Supported Models](#supported-models)
5. [Prompt Templates](#prompt-templates)
6. [Example](#example)
7. [License](#license)

---

## Key Features

- **Multiple Model Support**: Integrates easily with OpenAI API, local inference servers, and HuggingFace models.
- **Flexible Prompting System**: Choose from various prompt templates or design your own.
- **Batch Generation**: Efficiently generates claims for large datasets.
- **Knowledge Graph Integration**: Optional pipeline converting KG triples directly into natural language claims.
- **Configurable Generation Parameters**: Customize temperature, max length, batch size, and devices.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/claim-generator.git
cd claim-generator
```

Install using `pip`:

```bash
pip install .
```

Alternatively, you can manage dependencies manually:

```bash
pip install -r requirements.txt
```

Ensure your environment is set up with Python 3.10 or later.

---

## Quick Start

### Python API Usage

Here's a minimal example demonstrating how to use the `claim-generator` in Python:

```python
from claim_generator import create_generator, ModelConfig, ModelType, PromptTemplate

config = ModelConfig(
    model_type=ModelType.OPENAI,
    model_name_or_path="gpt-3.5-turbo",
    api_key="your-openai-api-key"
)

generator = create_generator(config, PromptTemplate.DEFAULT)

texts = [
    "NASA’s Perseverance rover discovered ancient microbial life on Mars.",
    "This finding changes our understanding of planetary exploration."
]

claims = generator.generate_claims(texts)
print(claims)
```

### CLI Usage

The CLI entry point is installed automatically and can be used as follows:

```bash
claim-generator \
  --model-type openai \
  --model-name-or-path gpt-3.5-turbo \
  --api-key your-openai-api-key \
  --input-file data/inputs/test_input.json \
  --output-file data/outputs/test_output.json
```

#### CLI Parameters

| Parameter               | Description                                      |
|-------------------------|--------------------------------------------------|
| `--model-type`          | Model type (`huggingface`, `openai`, `jan_local`). |
| `--model-name-or-path`  | Model identifier (OpenAI model or HuggingFace path). |
| `--api-key`             | API key for OpenAI or local inference server.    |
| `--endpoint-url`        | URL for local model inference server.            |
| `--prompt-template`     | Prompt template (`default`, `maximize_atomicity`, etc.). |
| `--temperature`         | Generation randomness. Default is `0.0`.         |
| `--max-length`          | Max input length. Default is `512`.              |
| `--batch-size`          | Batch size. Default is `8`.                      |
| `--input-file`          | Input JSON file path.                            |
| `--output-file`         | Output JSON file path.                           |

---

## Supported Models

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **HuggingFace**: Both seq2seq and causal models (e.g., T5, BART, GPT variants, LLaMA).
- **Local Servers**: Custom local inference endpoints compatible with OpenAI API structure.

---

## Prompt Templates

Choose from several built-in prompt strategies:

- **Default**: Balanced claim extraction.
- **Maximize Atomicity**: Claims split into smallest possible units.
- **Maximize Coverage**: Redundant claims to maximize factual coverage.
- **Granularity**: Adjust claim granularity (low, medium, high).

You can also define custom prompts easily.

---

## Example

Given the following input:

```json
[
  {
    "id": 1,
    "text": "NASA’s Perseverance rover discovered microbial life on Mars."
  }
]
```

Output JSON:

```json
[
  {
    "id": 1,
    "text": "NASA’s Perseverance rover discovered microbial life on Mars.",
    "claims": [
      "NASA’s Perseverance rover discovered microbial life.",
      "The discovery was made on Mars."
    ],
    "model_config": {
      "model_type": "openai",
      "model_name_or_path": "gpt-3.5-turbo",
      "device": "cpu",
      "temperature": 0.0,
      "max_length": 512,
      "batch_size": 8
    }
  }
]
```

---

## License

Licensed under the [MIT License](./LICENSE). You are free to modify, distribute, and use the code according to the license terms.
