def test_import():
    """测试包能否正常导入"""
    from dots_ocr_client import DotsOCRParser
    assert DotsOCRParser is not None

def test_init():
    """测试初始化"""
    from dots_ocr_client import DotsOCRParser
    parser = DotsOCRParser(ip="localhost", port=8000)
    assert parser.ip == "localhost"
    assert parser.port == 8000
