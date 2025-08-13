import io
import json
import os
from unittest.mock import patch, MagicMock, call
from PIL import Image
import tempfile
import fitz

from dots_ocr_client.parser import DotsOCRParser


def make_test_image(width=200, height=100):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return img


def fake_replicate_response():
    # Minimal layout output: one cell with a bbox and category and text
    return json.dumps([
        {"bbox": [10, 10, 100, 40], "category": "Text", "text": "hello"}
    ])


@patch("dots_ocr_client.model.inference.inference_with_replicate")
def test_parse_image_with_replicate_public(mock_infer):
    """Test basic replicate backend with public model"""
    mock_infer.return_value = fake_replicate_response()

    img = make_test_image()
    tmp_path = "./_tmp_unit_img.png"
    img.save(tmp_path)

    try:
        parser = DotsOCRParser(backend="replicate")
        results = parser.parse_image(tmp_path, "test", "prompt_layout_all_en", save_dir=None)
        assert isinstance(results, list) and len(results) == 1
        r = results[0]
        assert "cells" in r
        assert isinstance(r["cells"], list)
        assert r["cells"][0]["category"] == "Text"
        assert r["cells"][0]["bbox"]
        assert r["file_path"] == tmp_path
        
        # Verify inference_with_replicate was called with correct args
        mock_infer.assert_called_once()
        args, kwargs = mock_infer.call_args
        assert kwargs["deployment"] is None  # Public model
        assert kwargs["temperature"] == 0.1
        assert kwargs["top_p"] == 1.0
        assert kwargs["max_completion_tokens"] == 16384
        assert "api_token" in kwargs
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@patch("dots_ocr_client.model.inference.inference_with_replicate")
def test_parse_image_with_replicate_deployment(mock_infer):
    """Test replicate backend with custom deployment"""
    mock_infer.return_value = fake_replicate_response()

    img = make_test_image()
    tmp_path = "./_tmp_unit_img.png"
    img.save(tmp_path)

    try:
        parser = DotsOCRParser(
            backend="replicate",
            replicate_deployment="owner/custom-deployment"
        )
        results = parser.parse_image(tmp_path, "test", "prompt_layout_all_en", save_dir=None)
        assert isinstance(results, list) and len(results) == 1
        
        # Verify deployment was passed correctly
        mock_infer.assert_called_once()
        args, kwargs = mock_infer.call_args
        assert kwargs["deployment"] == "owner/custom-deployment"
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_completion_tokens"] == 16384
        assert "api_token" in kwargs
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def test_replicate_public_vs_deployment():
    """Test that public model uses run() and deployment uses deployments.get()"""
    with patch("replicate.run") as mock_run, \
         patch("replicate.deployments.get") as mock_get:
        
        from dots_ocr_client.model.inference import inference_with_replicate
        
        # Setup mock for public model
        mock_run.return_value = json.dumps([{"bbox": [1, 2, 3, 4], "category": "Text", "text": "test"}])
        
        # Setup mock for deployment
        mock_deployment = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.output = json.dumps([{"bbox": [1, 2, 3, 4], "category": "Text", "text": "test"}])
        mock_prediction.wait = MagicMock()
        mock_deployment.predictions.create.return_value = mock_prediction
        mock_get.return_value = mock_deployment
        
        img = make_test_image()
        
        # Test public model
        result = inference_with_replicate(img, "test prompt", deployment=None)
        mock_run.assert_called_once_with("sljeff/dots.ocr", input={
            "image": mock_run.call_args[1]["input"]["image"],
            "prompt": "test prompt",
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 2048,
        })
        
        # Test custom deployment
        mock_run.reset_mock()
        mock_get.reset_mock()
        result = inference_with_replicate(img, "test prompt", deployment="owner/model")
        mock_get.assert_called_once_with("owner/model")
        mock_deployment.predictions.create.assert_called_once()


@patch("dots_ocr_client.model.inference.inference_with_replicate")
def test_parse_pdf_with_replicate_multipage(mock_infer):
    """Test multi-page PDF processing with replicate backend"""
    
    # Create responses for 3 pages
    page_responses = [
        json.dumps([{"bbox": [10, 10, 100, 40], "category": "Title", "text": f"Page {i+1}"}])
        for i in range(3)
    ]
    mock_infer.side_effect = page_responses
    
    # Create a test PDF with 3 pages
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), f"Page {i+1}")
        doc.save(pdf_path)
        doc.close()
    
    try:
        parser = DotsOCRParser(backend="replicate", num_thread=2)
        results = parser.parse_pdf(pdf_path, "test.pdf", "prompt_layout_all_en", save_dir=None)
        
        # Should have 3 pages of results
        assert len(results) == 3
        # Sort results by page_no since multi-threading may return them out of order
        results_sorted = sorted(results, key=lambda x: x["page_no"])
        for i, result in enumerate(results_sorted):
            assert result["page_no"] == i
            assert result["cells"][0]["text"] == f"Page {i+1}"
        
        # Verify inference was called 3 times
        assert mock_infer.call_count == 3
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)



