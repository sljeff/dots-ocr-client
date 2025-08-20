def test_import():
    """测试包能否正常导入"""
    from dots_ocr_client import DotsOCRParser
    assert DotsOCRParser is not None

def test_init():
    """测试初始化"""
    from dots_ocr_client import DotsOCRParser
    parser = DotsOCRParser(backend="vllm", base_url="http://localhost:8000", api_token="test")
    assert parser.backend == "vllm"
    assert parser.base_url == "http://localhost:8000"
    assert parser.api_token == "test"
