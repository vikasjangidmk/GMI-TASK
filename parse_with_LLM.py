# ✅ Updated parse_with_LLM.py — fixed type issue in postprocess_task3 and improved robustness

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

load_dotenv()

def handle_json(response_text):
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return response_text[start:end]
    except Exception:
        return response_text

def parse_structured_data(input_text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a financial document parser.\n"
        "Given the OCR text below from a scanned bank statement, extract ONLY the following fields in JSON format:\n"
        "- header: {account_holder, bank, date_range}\n"
        "- transactions: array of {date, description, amount, balance}\n"
        "- summary: {final_balance, total_debit, total_credit}\n\n"
        "Notes:\n"
        "- Only include transactions that look valid with dates like DD/MM/YYYY or similar.\n"
        "- Use empty strings for missing fields, don't return 'Unavailable'.\n"
        "- Respond with a clean JSON, no extra commentary.\n\n"
        f"OCR Text:\n{input_text}"
    )

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial statement parser."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        raw_response = result.choices[0].message.content
        return json.loads(handle_json(raw_response))
    except Exception as e:
        print("❌ Error parsing structured data:", e)
        return {}

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
        txn["date"] = fix_ocr_text(str(txn.get("date", "")))
        txn["description"] = fix_ocr_text(str(txn.get("description", "")))
        txn["amount"] = fix_ocr_text(str(txn.get("amount", "")))
        txn["balance"] = fix_ocr_text(str(txn.get("balance", "")))
        if not is_valid_date(txn["date"]):
            txn["date_valid"] = False
        if not is_valid_amount(txn["amount"]):
            txn["amount_valid"] = False
    return data

def export_table_to_excel_openpyxl(table_data, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"

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
    print(f"📁 Excel file saved to: {output_path}")
