"""Test the data structure returned by dots-ocr-client"""
import json
from pprint import pprint
from dots_ocr_client import DotsOCRParser

def test_mock_data_structure():
    """Test with mock VLLM server to check data structure"""
    # This test will fail if no VLLM server is running
    # It's mainly for documenting the expected structure
    
    # Expected structure for a single page
    expected_structure = {
        'page_no': int,  # Page number (0-based)
        'file_path': str,  # Original file path
        'input_height': int,  # Processed image height
        'input_width': int,  # Processed image width
        'cells': [  # List of detected layout elements
            {
                'bbox': [int, int, int, int],  # [x1, y1, x2, y2]
                'category': str,  # 'Text', 'Title', 'Table', etc.
                'text': str  # OCR result (optional for some categories)
            }
        ],
        'image_with_layout': object,  # PIL.Image object (optional)
        'md_content': str,  # Markdown formatted content
        'md_content_no_hf': str,  # Markdown without header/footer
    }
    
    print("Expected data structure:")
    print(json.dumps(expected_structure, default=str, indent=2))
    
def inspect_real_structure():
    """Inspect the actual structure with a real PDF (requires VLLM server)"""
    import os
    
    pdf_path = os.path.join(os.path.dirname(__file__), 'test_document.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"Test PDF not found at {pdf_path}")
        return
    
    print(f"\nTesting with PDF: {pdf_path}")
    print("Note: This requires a running VLLM server at localhost:8000")
    print("-" * 60)
    
    try:
        # Initialize parser (assuming local VLLM server)
        parser = DotsOCRParser(
            ip="localhost",
            port=8000,
            num_thread=1
        )
        
        # Parse just the first page to see structure
        results = parser.parse_pdf(
            pdf_path,
            filename="test_document",
            prompt_mode="prompt_layout_all_en",
            save_dir=None
        )
        
        if results:
            print(f"\nNumber of pages processed: {len(results)}")
            print("\nStructure of first page result:")
            print("-" * 60)
            
            # Show keys and types
            first_page = results[0]
            for key, value in first_page.items():
                if key == 'cells' and isinstance(value, list) and value:
                    print(f"  '{key}': list of {len(value)} items")
                    print(f"    First cell structure:")
                    for cell_key, cell_value in value[0].items():
                        value_repr = f"{type(cell_value).__name__}"
                        if isinstance(cell_value, str):
                            value_repr += f" (length: {len(cell_value)})"
                        elif isinstance(cell_value, list):
                            value_repr += f" {cell_value}"
                        print(f"      '{cell_key}': {value_repr}")
                elif key == 'image_with_layout':
                    print(f"  '{key}': {type(value).__name__}")
                elif isinstance(value, str):
                    print(f"  '{key}': str (length: {len(value)})")
                else:
                    print(f"  '{key}': {type(value).__name__} = {value}")
            
            print("\nSample cells (first 3):")
            print("-" * 60)
            for i, cell in enumerate(first_page.get('cells', [])[:3]):
                print(f"Cell {i}:")
                print(f"  Category: {cell.get('category')}")
                print(f"  BBox: {cell.get('bbox')}")
                text = cell.get('text', '')
                if text:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"  Text: {preview}")
                print()
                
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the VLLM server is running at localhost:8000")
        print("You can also modify the ip/port in the test if needed")

if __name__ == "__main__":
    test_mock_data_structure()
    print("\n" + "=" * 60)
    inspect_real_structure()
