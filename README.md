# dots-ocr-client

A Python client library for dots.ocr VLLM service - forked from the original dots.ocr with minimal dependencies and no file I/O.

## Why this fork?

This is a client-only version of [dots.ocr](https://github.com/rednote-hilab/dots.ocr) that removes all server-side dependencies (transformers, flash-attn, etc.), reducing the package size from several GB to just a few MB. Perfect for production environments where you only need to call an already-deployed dots.ocr VLLM service.

## Installation

```bash
# Install directly from GitHub
uv add git+https://github.com/sljeff/dots-ocr-client.git

# Or with pip
pip install git+https://github.com/sljeff/dots-ocr-client.git
```

## Prerequisites

You need to have a dots.ocr VLLM service already deployed and accessible. See the [original repository](https://github.com/rednote-hilab/dots.ocr) for deployment instructions.

## Usage

```python
from dots_ocr_client import DotsOCRParser

# Initialize the client (pointing to your VLLM service)
parser = DotsOCRParser(
    ip="localhost",  # Your VLLM server IP
    port=8000,       # Your VLLM server port
    num_thread=16    # Number of threads for parallel processing
)

# Parse a PDF document
results = parser.parse_pdf(
    "document.pdf",
    prompt_mode="prompt_layout_all_en"  # Extract layout, text, tables, formulas
)

# Process results
for page in results:
    print(f"Page {page['page_no']}:")
    for cell in page['cells']:
        print(f"  - {cell['category']}: {cell.get('text', '')[:50]}...")
```

## Differences from Original dots.ocr

### Removed
- ❌ Local model inference (transformers, flash-attn)
- ❌ Gradio/Streamlit demos
- ❌ Docker configurations
- ❌ Model training/evaluation tools
- ❌ Heavy ML dependencies (~3GB)
- ❌ Temporary file I/O operations

### Kept
- ✅ VLLM API client functionality
- ✅ PDF/Image processing
- ✅ Layout detection and OCR
- ✅ Multi-threading support
- ✅ Output cleaning and formatting
- ✅ Coordinate transformation

### Dependencies Comparison
| Original dots.ocr | dots-ocr-client |
|------------------|-----------------|
| transformers (500MB+) | ❌ Removed |
| flash-attn (requires compilation) | ❌ Removed |
| torch/accelerate (2GB+) | ❌ Removed |
| numpy (60MB+ installed) | ❌ Removed (was unused) |
| gradio | ❌ Removed |
| qwen_vl_utils | ❌ Removed |
| tqdm | ❌ Removed |
| PyMuPDF (53MB) | ✅ Kept |
| Pillow (13MB) | ✅ Kept |
| openai (6.6MB) | ✅ Kept |
| pydantic (7.3MB) | ✅ Kept |
| requests (0.4MB) | ✅ Kept (for image downloads) |
| **Total venv Size** | **2-3GB+ → 96MB** (actual measured) |

## Data Structure

The SDK returns a list of dictionaries, where each dictionary represents one page of the parsed document:

```python
[
    {
        'page_no': 0,                    # Page number (0-based index)
        'file_path': 'document.pdf',     # Original file path
        'input_height': 2212,             # Processed image height
        'input_width': 1708,              # Processed image width
        
        # Core data: detected layout elements
        'cells': [
            {
                'bbox': [41, 589, 103, 1587],  # Bounding box [x1, y1, x2, y2]
                'category': 'Text',             # Element type
                'text': 'Extracted text...'     # OCR result
            },
            {
                'bbox': [167, 323, 1486, 464],
                'category': 'Title',
                'text': 'Document Title Here'
            },
            # ... more cells
        ],
        
        # Additional outputs
        'image_with_layout': PIL.Image,   # Image with layout boxes drawn
        'md_content': '# Title\n...',     # Full Markdown content
        'md_content_no_hf': '...',        # Markdown without headers/footers
    },
    # ... more pages
]
```

### Cell Categories

Each cell in the `cells` list has a `category` field that indicates the type of layout element:

- **Text** - Regular text paragraphs
- **Title** - Document or section titles
- **Table** - Table content
- **Picture** - Images (no text field)
- **Formula** - Mathematical formulas
- **Section-header** - Section headings
- **List-item** - Bulleted or numbered list items
- **Caption** - Figure/table captions
- **Footnote** - Page footnotes
- **Page-header** / **Page-footer** - Headers and footers
- **Other** / **Unknown** - Unclassified elements

## API Reference

### DotsOCRParser

Main class for document parsing.

**Parameters:**
- `ip` (str): VLLM server IP address
- `port` (int): VLLM server port
- `num_thread` (int): Number of threads for parallel processing
- `dpi` (int): DPI for PDF rendering (default: 200)
- `temperature` (float): Model temperature (default: 0.1)

**Methods:**
- `parse_pdf(pdf_path, prompt_mode)`: Parse a PDF file
- `parse_image(image_path, prompt_mode)`: Parse a single image
- `parse_file(file_path, prompt_mode)`: Auto-detect and parse file

## License

Same as original dots.ocr - [LICENSE](LICENSE)

## Credits

This is a fork of [dots.ocr](https://github.com/rednote-hilab/dots.ocr) by HiLab. All core algorithms and model training work credit goes to the original authors.
