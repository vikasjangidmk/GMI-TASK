import itertools
from operator import itemgetter
import pytesseract
import tiktoken
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vikas\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def num_tokens(text, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoding.encode(text))
    return num_tokens

def limit_tokens(text, max_tokens=16000):
    num_of_tokens = num_tokens(text)
    if num_of_tokens > max_tokens:
        return text[:max_tokens]
    else:
        return text

def cluster_list(xs, tolerance=0):
    if tolerance == 0 or len(xs) < 2:
        return [[x] for x in sorted(xs)]
    
    groups = []
    xs = list(sorted(xs))
    current_group = [xs[0]]
    last = xs[0]

    for x in xs[1:]:
        if x <= (last + tolerance):
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]
        last = x
    groups.append(current_group)

    return groups

def make_cluster_dict(values, tolerance):
    clusters = cluster_list(list(set(values)), tolerance)

    nested_tuples = [
        [(val, i) for val in value_cluster] for i, value_cluster in enumerate(clusters)
    ]

    return dict(itertools.chain(*nested_tuples))

def cluster_objects(xs, tolerance):
    key_fn = lambda x: (x['coordinates'][1] + x['coordinates'][3]) / 2
    values = map(key_fn, xs)
    cluster_dict = make_cluster_dict(values, tolerance)
    get_0, get_1 = itemgetter(0), itemgetter(1)
    cluster_tuples = sorted(((x, cluster_dict.get(key_fn(x))) for x in xs), key=get_1)
    grouped = itertools.groupby(cluster_tuples, key=get_1)

    return [list(map(get_0, v)) for k, v in grouped]

def get_avg_char_width(data):
    height = 1000
    sum_widths = 0.0
    cnt = 0
    for datum in data:
        height = min(height, abs(datum['coordinates'][3] - datum['coordinates'][1]))
        sum_widths += datum['coordinates'][2] - datum['coordinates'][0]
        cnt += len(datum['value'])
    return height / 2, sum_widths // cnt

def collate_line(line_chars, tolerance, add_spaces) -> str:
    coll = ""
    last_x1 = 0

    for char in sorted(line_chars, key=lambda x: x['coordinates'][0]):
        coll += ' '
        last_x1 += tolerance
        while last_x1 + tolerance < char['coordinates'][0] and add_spaces: 
            coll += " "
            last_x1 += tolerance
        coll += char['value']
        last_x1 = char['coordinates'][2]

    return coll[1:] if add_spaces else coll.strip()

def extract_text(data, add_spaces, max_tokens=16000):
    min_height, x_tolerance = get_avg_char_width(data)
    
    doctop_clusters = cluster_objects(data, tolerance=min_height)

    lines = (
        collate_line(line_chars, x_tolerance, add_spaces) for line_chars in doctop_clusters
    )

    text = "\n".join(lines)

    return limit_tokens(text, max_tokens)

def extract_text_ocr(image_file, add_spaces, max_tokens=16000):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = Image.open(image_file)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    data = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():  
            datum = {
                'value': ocr_data['text'][i],
                'coordinates': [
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i]
                ]
            }
            data.append(datum)
    return extract_text(data, add_spaces, max_tokens)

# Example usage:
# image = Image.open(r'C:\Users\Jawahar\Documents\Interview_task\GMIndia\ocr_check\AOUT 2021.pdf0.jpg')
# extracted_text = extract_text_ocr(image, add_spaces=True, max_tokens=16000)

#save the extracted text
# with open(r'C:\Users\Jawahar\Documents\Interview_task\GMIndia\ocr_check\extracted_text.txt', 'w') as text_file:
# 	text_file.write(extracted_text)
# print(extracted_text)