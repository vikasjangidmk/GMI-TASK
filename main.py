# ✅ FINAL main.py — Fully Updated for image/pdf input → OCR → LLM → JSON + Excel

import os
import json
import cv2
import unicodedata
import re
from dotenv import load_dotenv
from preprocess import preprocess_document
from extract_ocr import extract_text_ocr
from extract_pdf import extract_text_pdf
from parse_with_LLM import (
    parse_structured_data,
    postprocess_task3,
    export_table_to_excel_openpyxl
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def ensure_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY is not set.")
    os.environ["OPENAI_API_KEY"] = api_key

def slugify_filename(filename):
    nfkd = unicodedata.normalize('NFKD', filename)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r'[^\w\-_. ]', '_', ascii_str)

def process_file(input_path, output_dir, prompt, add_spaces=True, lang='en'):
    ensure_api_key()
    file_ext = os.path.splitext(input_path)[1].lower()
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clean_name = slugify_filename(base_name)

    # STEP 1: Extract text
    if file_ext in [".jpg", ".jpeg", ".png"]:
        print("🖼️ Image detected. Running preprocessing + OCR...")
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Failed to read image: {input_path}")
            return

        angle, corrected_image = preprocess_document(image)
        print(f"✅ Skew corrected. Angle: {angle:.2f}°")

        corrected_dir = os.path.join(output_dir, "corrected_images")
        os.makedirs(corrected_dir, exist_ok=True)
        corrected_image_path = os.path.join(corrected_dir, f"corrected_{clean_name}.jpg")
        cv2.imwrite(corrected_image_path, corrected_image)
        print(f"📷 Corrected image saved at: {corrected_image_path}")

        extracted_text = extract_text_ocr(corrected_image_path, add_spaces=add_spaces, max_tokens=16000, lang=lang)

    elif file_ext == ".pdf":
        print("📄 PDF detected. Extracting text from PDF...")
        extracted_text = extract_text_pdf(input_path, multiple_pages=True, max_page_count=3, max_tokens=16000, lang=lang)

    else:
        print(f"❌ Unsupported file type: {file_ext}")
        return

    if not extracted_text.strip():
        print("⚠️ No text extracted. Skipping file.")
        return

    print("\n🔍 Extracted Text Preview (first 500 chars):\n")
    print(extracted_text[:500])

    debug_text_path = os.path.join(output_dir, f"{clean_name}_ocr_debug.txt")
    with open(debug_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"📝 Full OCR text saved for debugging: {debug_text_path}")

    # STEP 2: GPT Parsing
    enhanced_prompt = (
        "You are an intelligent bank statement parser. "
        "Your task is to extract structured data from scanned or OCR'd text. "
        "Return the output in the following JSON structure:\n\n"
        "{\n"
        "  \"header\": {\"account_holder\": \"\", \"bank\": \"\", \"date_range\": \"\"},\n"
        "  \"transactions\": [\n"
        "    {\"date\": \"\", \"description\": \"\", \"amount\": \"\", \"balance\": \"\"},\n"
        "    ...\n"
        "  ],\n"
        "  \"summary\": {\"final_balance\": \"\", \"total_debit\": \"\", \"total_credit\": \"\"}\n"
        "}\n\n"
        "Make your best guess even if some information is unclear or approximate."
        " Use the following OCR text as source:\n\n"
    )

    try:
        parsed_json = parse_structured_data(enhanced_prompt + extracted_text)
        parsed_json = postprocess_task3(parsed_json)
    except Exception as e:
        print(f"❌ GPT parsing failed: {e}")
        return

    # STEP 3: Save JSON
    json_output_dir = os.path.join(output_dir, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    json_path = os.path.join(json_output_dir, f"{clean_name}_parsed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=4, ensure_ascii=False)
    print(f"📝 JSON saved to: {json_path}")

    # STEP 4: Save Excel
    excel_output_dir = os.path.join(output_dir, "excel")
    os.makedirs(excel_output_dir, exist_ok=True)
    excel_path = os.path.join(excel_output_dir, f"{clean_name}_parsed.xlsx")
    try:
        table_data = {
            "columns": ["Date", "Description", "Amount", "Balance"],
            "rows": [
                [txn.get("date", ""), txn.get("description", ""), txn.get("amount", ""), txn.get("balance", "")]
                for txn in parsed_json.get("transactions", [])
                if txn.get("date") and txn.get("amount")
            ]
        }
        export_table_to_excel_openpyxl(table_data, excel_path)
        print(f"📊 Excel saved to: {excel_path}")
    except Exception as e:
        print(f"❌ Excel export failed: {e}")

    print("✅ All tasks completed successfully!")

# -------------------- Run Script --------------------
if __name__ == "__main__":
    input_path = r"C:\Users\vikas\OneDrive\Desktop\GMI-TASK\gmindia-challlenge-012024-datas\banquepopulaire\EXTRAIT-22217785648-20191031.pdf_1.jpg"
    output_directory = r"C:\\Users\\vikas\\OneDrive\\Desktop\\GMI-TASK\\output"
    prompt = ""  # prompt now embedded in the enhanced_prompt above
    process_file(input_path, output_directory, prompt, lang='en')
