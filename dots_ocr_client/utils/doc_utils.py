import pypdfium2 as pdfium
import enum
from pydantic import BaseModel, Field
from PIL import Image


class SupportedPdfParseMethod(enum.Enum):
    OCR = 'ocr'
    TXT = 'txt'


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')


def pdfium_doc_to_image(page, target_dpi=200, origin_dpi=None) -> Image.Image:
    """Convert pypdfium2 page to PIL Image.

    Args:
        page: pypdfium2 page object
        target_dpi (int, optional): target DPI for rendering. Defaults to 200.
        origin_dpi: unused, kept for compatibility

    Returns:
        PIL.Image: rendered image
    """
    
    # Calculate scale factor based on target DPI
    scale = target_dpi / 72.0
    
    # Render page to bitmap
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    
    # Handle large images by reducing DPI
    if pil_image.width > 4500 or pil_image.height > 4500:
        scale = 72.0 / 72.0  # Use default DPI
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
    
    return pil_image.convert('RGB')


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
    doc = pdfium.PdfDocument(pdf_file)
    try:
        pdf_page_num = len(doc)
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            print('end_page_id is out of range, use images length')
            end_page_id = pdf_page_num - 1

        for index in range(start_page_id, end_page_id + 1):
            page = doc[index]
            img = pdfium_doc_to_image(page, target_dpi=dpi)
            images.append(img)
    finally:
        doc.close()
    return images

