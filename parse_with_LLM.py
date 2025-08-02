import os
import json
import easyocr
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

# ✅ Load OpenAI API key from .env
load_dotenv()

def handle_json(response_text):
    """Safely extract JSON block from GPT output."""
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return response_text[start:end]
    except Exception:
        return response_text  # fallback

# ----------------- Task 3 ------------------
def parse_structured_data(input_text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a bank statement parser.\n"
        "Given the OCR text below from a scanned bank statement, convert it into structured JSON with 3 parts:\n"
        "1. header (bank info, date range)\n"
        "2. transactions (date, description, amount, balance)\n"
        "3. summary (totals, loyalty, footer info)\n"
        "Only respond in this format:\n"
        "{\n"
        "  \"header\": [...],\n"
        "  \"transactions\": [ {\"date\": \"\", \"description\": \"\", \"amount\": \"\", \"balance\": \"\" }, ...],\n"
        "  \"summary\": [...]\n"
        "}\n\n"
        f"OCR Text:\n{input_text}"
    )

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial document parser."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        raw_response = result.choices[0].message.content
        return json.loads(handle_json(raw_response))

    except Exception as e:
        print("❌ Error parsing structured data (Task 3):", e)
        return {}

# ----------------- Task 4 ------------------
def parse_tabular_data(input_text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a tabular data extractor.\n"
        "Given OCR text from a scanned financial document, detect and reconstruct tables accurately.\n"
        "Your output should be:\n"
        "{\n"
        "  \"columns\": [\"Date\", \"Description\", \"Amount\", \"Balance\"],\n"
        "  \"rows\": [\n"
        "    [\"30/03/2022\", \"FREE MOBILE\", \"-29.99\", \"\"],\n"
        "    [\"31/03/2022\", \"ATM Withdrawal\", \"-100.00\", \"1714.96\"]\n"
        "  ]\n"
        "}\n\n"
        f"OCR Text:\n{input_text}"
    )

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial table parser."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        raw_response = result.choices[0].message.content
        return json.loads(handle_json(raw_response))

    except Exception as e:
        print("❌ Error parsing table data (Task 4):", e)
        return {}

# ----------------- Task 5 ------------------
def fix_ocr_text(text):
    corrections = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1',
        ',': '.', '|': ''
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text.strip()

def is_valid_date(date_str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except:
            continue
    return False

def is_valid_amount(value):
    try:
        float(value.replace(",", "."))
        return True
    except:
        return False

def postprocess_task3(data):
    for txn in data.get("transactions", []):
        txn["date"] = fix_ocr_text(txn.get("date", ""))
        txn["description"] = fix_ocr_text(txn.get("description", ""))
        txn["amount"] = fix_ocr_text(txn.get("amount", ""))
        txn["balance"] = fix_ocr_text(txn.get("balance", ""))
        if not is_valid_date(txn["date"]):
            txn["date_valid"] = False
        if not is_valid_amount(txn["amount"]):
            txn["amount_valid"] = False
    return data

def postprocess_task4(data):
    cleaned_rows = []
    for row in data.get("rows", []):
        if len(row) != 4:
            continue
        date, desc, amt, bal = map(fix_ocr_text, row)
        row_data = [date, desc, amt, bal]
        if is_valid_date(date) and is_valid_amount(amt):
            cleaned_rows.append(row_data)
    data["rows"] = cleaned_rows
    return data

# ----------------- Task 6 ------------------
def export_table_to_excel_openpyxl(table_data, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"

    # Headers
    headers = table_data.get("columns", ["Date", "Description", "Amount", "Balance"])
    ws.append(headers)

    for col_num, col in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for row in table_data.get("rows", []):
        ws.append(row)

    for column_cells in ws.columns:
        length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length + 2

    wb.save(output_path)
    print(f"📁 Excel file (Task 6) saved to: {output_path}")

# ----------------- MAIN ------------------
if __name__ == "__main__":
    image_path = r"C:\Users\vikas\OneDrive\Desktop\GMI-TASK\gmindia-challlenge-012024-datas\societegenerale\image12022.jpg"
    output_folder = r"C:\Users\vikas\OneDrive\Desktop\GMI-TASK\output\ocr"
    ocr_text_file = os.path.join(output_folder, "ocr_output.txt")
    structured_json = os.path.join(output_folder, "task3_parsed_statement.json")
    table_json = os.path.join(output_folder, "task4_table_parsed.json")
    excel_output_path = os.path.join(output_folder, "task6_table_output.xlsx")

    os.makedirs(output_folder, exist_ok=True)

    print("🔍 Performing OCR...")
    reader = easyocr.Reader(['fr'], gpu=False)
    results = reader.readtext(image_path)
    ocr_text = "\n".join([text for _, text, conf in results if conf > 0.5])

    with open(ocr_text_file, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    print("🧠 Task 3: Structured parsing...")
    structured_result = parse_structured_data(ocr_text)
    with open(structured_json, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=4, ensure_ascii=False)

    print("📊 Task 4: Table parsing...")
    table_result = parse_tabular_data(ocr_text)
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(table_result, f, indent=4, ensure_ascii=False)

    print("🛠️ Post-processing Task 3...")
    structured_result = postprocess_task3(structured_result)
    with open(structured_json, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=4, ensure_ascii=False)

    print("🛠️ Post-processing Task 4...")
    table_result = postprocess_task4(table_result)
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(table_result, f, indent=4, ensure_ascii=False)

    print("📤 Task 6: Exporting to Excel...")
    export_table_to_excel_openpyxl(table_result, excel_output_path)

    print("\n✅ All Tasks (1-6) Completed Successfully!")
