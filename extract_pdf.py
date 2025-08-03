import os
import pypdfium2 as pdfium
import tiktoken
from extract_ocr import extract_text_ocr

def num_tokens(text, model="gpt-3.5-turbo-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def limit_tokens(text, max_tokens=16000):
    return text[:max_tokens] if num_tokens(text) > max_tokens else text

def extract_text_pdf(feed: str, multiple_pages: bool = False, max_page_count: int = 3,
                     page_num: int = 1, max_tokens: int = 16000, lang: str = 'en') -> str:
    """
    Extracts text from a PDF, using OCR as fallback if necessary.
    """
    try:
        pdf = pdfium.PdfDocument(feed)
        texts = []

        if multiple_pages:
            for i in range(min(len(pdf), max_page_count)):
                text = _get_text_with_fallback(pdf[i], feed, i + 1, lang)
                texts.append(text)
        else:
            text = _get_text_with_fallback(pdf[page_num - 1], feed, page_num, lang)
            texts.append(text)

        del pdf  # release PDF handle
        return limit_tokens("\n".join(texts), max_tokens)

    except Exception as e:
        print(f"❌ Failed to process PDF: {e}")
        return ""

def _get_text_with_fallback(page, pdf_path, page_number: int, lang: str) -> str:
    try:
        text = page.get_textpage().get_text_range()
        if text and any(c.isalnum() for c in text):
            return text
    except:
        pass

    # Fallback: render as image and OCR it
    try:
        image = page.render(scale=3).to_pil()
        image_path = os.path.splitext(pdf_path)[0] + f"_page_{page_number}.jpg"
        image.save(image_path)
        return extract_text_ocr(image_path, add_spaces=True, max_tokens=16000, lang=lang)
    except Exception as e:
        print(f"❌ OCR fallback failed on page {page_number}: {e}")
        return ""
