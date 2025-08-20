# dots-ocr-client

A lightweight Python client for dots.ocr. Call either a self-hosted vLLM service or Replicate with the same API. No heavy deps, no file I/O. Based on the original project [dots.ocr](https://github.com/rednote-hilab/dots.ocr).

## Usage

### Use with [Replicate](https://replicate.com/sljeff/dots.ocr)

- First, get your [API token](https://replicate.com/account/api-tokens).

```python
from dots_ocr_client.parser import DotsOCRParser

# Default: use the public model sljeff/dots.ocr
parser = DotsOCRParser(
    backend="replicate",
    api_token="your-replicate-token"  # Required
)
results = parser.parse_file("/path/to/file.pdf", prompt_mode="prompt_layout_all_en")
```

<details>
<summary>Advanced: use your own Replicate deployment</summary>

For better performance and dedicated resources, you can create your own deployment:

1. **Create a deployment on Replicate:**
   - Go to your [Replicate Deployments page](https://replicate.com/deployments)
   - Click "Create deployment"
   - Select model: `sljeff/dots.ocr`
   - Choose your hardware configuration
   - Name your deployment (e.g., `yourname/dots-ocr`)

2. **Get your API token** from https://replicate.com/account/api-tokens

3. **Use your deployment in code:**
```python
from dots_ocr_client.parser import DotsOCRParser

parser = DotsOCRParser(
    backend="replicate",
    api_token="your-api-token",               # Required
    replicate_deployment="yourname/dots-ocr",  # your deployment name
)
results = parser.parse_file("/path/to/file.pdf", prompt_mode="prompt_layout_all_en")
```

</details>

### Use with vLLM

Prerequisite: have a running dots.ocr vLLM service.

```python
from dots_ocr_client.parser import DotsOCRParser

parser = DotsOCRParser(
    backend="vllm",
    base_url="http://localhost:8000",  # Your vLLM server URL
    api_token="your-api-token",       # Optional, depends on your setup
    model_name="model",
)
results = parser.parse_file("/path/to/file.pdf", prompt_mode="prompt_layout_all_en")
```

## Installation

Install this project directly from Git.

With uv:
```bash
uv add git+https://github.com/sljeff/dots-ocr-client.git
```

With pip:
```bash
pip install git+https://github.com/sljeff/dots-ocr-client.git
```

## Why this fork & Differences

This is a client-only fork focusing on:
- Minimal dependencies (no transformers/flash-attn, etc.)
- Simple API to call existing deployments (vLLM or Replicate)
- No file outputs; functions return in-memory results

## API Reference

Constructor:
```python
DotsOCRParser(
  backend: str = "vllm",                   # "vllm" or "replicate"
  base_url: str = "http://127.0.0.1:8000", # for vLLM backend
  api_token: str | None = None,            # API token for both backends
  model_name: str = "model",
  temperature: float = 0.1,
  top_p: float = 1.0,
  max_completion_tokens: int = 16384,
  num_thread: int = 64,
  dpi: int = 200,
  min_pixels: int | None = None,
  max_pixels: int | None = None,
  replicate_deployment: str | None = None, # if None and backend=replicate -> public model sljeff/dots.ocr
)
```

Methods:
- `parse_file(path, prompt_mode="prompt_layout_all_en", bbox=None, fitz_preprocess=False)`
- `parse_pdf(input_path, filename, prompt_mode, save_dir)`
- `parse_image(input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False)`

The sampling parameters `temperature`, `top_p`, and `max_completion_tokens` have the same meaning on both backends.

## Data Structure

The SDK returns a list of dictionaries, where each dictionary represents one page:

```python
[
  {
    "page_no": 0,
    "file_path": "document.pdf",
    "input_height": 2212,
    "input_width": 1708,

    # Core data: detected layout elements
    "cells": [
      {
        "bbox": [41, 589, 103, 1587],
        "category": "Text",
        "text": "Extracted text..."
      },
      {
        "bbox": [167, 323, 1486, 464],
        "category": "Title",
        "text": "Document Title Here"
      }
      # ...
    ],

    # Additional outputs
    "image_with_layout": "<PIL.Image>",
    "md_content": "# Title\n...",
    "md_content_no_hf": "..."
  },
  # ... more pages
]
```

Common categories include: Text, Title, Table, Picture, Formula, Section-header, List-item, Caption, Footnote, Page-header, Page-footer, Other, Unknown.

## License

See LICENSE and NOTICE.
