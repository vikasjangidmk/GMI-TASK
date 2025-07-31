import pypdfium2 as pdfium 
import tiktoken

def num_tokens(text, model="gpt-3.5-turbo-0613"):
	"""Return the number of tokens used by a list of messages."""
	try:
		encoding = tiktoken.encoding_for_model(model)
	except KeyError:
		print("Warning: model not found. Using cl100k_base encoding.")
		encoding = tiktoken.get_encoding("cl100k_base")

	num_tokens = len(encoding.encode(text))
	return num_tokens


def limit_tokens (text, max_tokens=16000):
	num_of_tokens = num_tokens(text)
	if num_of_tokens > max_tokens:
		return text[:max_tokens]
	else:
		return text

def extract_text_pdf(feed: str, multiple_pages: bool = False, max_page_count: int=2, page_num: int = 1, max_tokens: int = 16000) -> str:
	""" 	This function makes use of the PyPDFium2 library to extract the text from a pdf file	"""
	if multiple_pages == False:
		pdf = pdfium.PdfDocument(feed)
		text = pdf[page_num - 1].get_textpage().get_text_range()
		return limit_tokens (text, max_tokens=max_tokens)
	else:
		data = []
		pdf = pdfium.PdfDocument(feed)
		for i in range (min(len(pdf), max_page_count)):
			data.append (pdf[i].get_textpage().get_text_range())
		text = "\n".join(data)
		return limit_tokens (text, max_tokens=max_tokens)

# text = extract_text_pdf(r"C:\Users\Jawahar\Downloads\Jawahar_C_Resume.pdf", multiple_pages=True, max_page_count=3, max_tokens=16000)
# print(text)