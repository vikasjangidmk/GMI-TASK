import os
import json
import cv2
from preprocess import correct_skew
from extract_ocr import extract_text_ocr
from parse_with_LLM import parse_with_gpt

def ensure_api_key():
    """
    Ensure the OpenAI API key is set in the environment.
    If not, prompt the user or handle securely.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Please set it before running the program.")
    
    os.environ["OPENAI_API_KEY"] = api_key

def process_image(image_path, output_dir, prompt, add_spaces=True, ocr=True):
    """Processes the image to extract text, sends it to GPT for parsing, and saves the result as a JSON file."""
    ensure_api_key()
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    angle, corrected_image = correct_skew(image)
    print(f"Corrected skew angle: {angle}")

    corrected_image_dir = os.path.join(output_dir, 'corrected_images')
    os.makedirs(corrected_image_dir, exist_ok=True)

    corrected_image_path = os.path.join(corrected_image_dir, f"corrected_{os.path.basename(image_path)}")
    cv2.imwrite(corrected_image_path, corrected_image)
    print(f"Corrected image saved to {corrected_image_path}")

    with open(corrected_image_path, 'rb') as img_file:
        extracted_text = extract_text_ocr(img_file, add_spaces, max_tokens=16000)

        if not extracted_text.strip():
            print(f"OCR failed or no text extracted from {image_path}. Skipping...")
            return

        prompt_ending = '\n' if extracted_text[-1] != '\n' else ''
        full_prompt = prompt + '\n' + extracted_text + prompt_ending

        try:
            gpt_response = parse_with_gpt(full_prompt)
        except Exception as e:
            print(f"Error with GPT API: {e}")
            return

        base_name = os.path.basename(image_path)
        output_file_name = f"{os.path.splitext(base_name)[0]}.json"
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, 'w') as json_file:
            json.dump(gpt_response, json_file, indent=4)

        print(f"Output saved to {output_file_path}")

if __name__ == "__main__":
    # Example usage
    input_image = r'C:\Users\Jawahar\Documents\Interview_task\GMIndia\gmindia-challlenge-012024-datas\laposte\releve_CCP5217718Y033_20191129_page-0001.jpg'
    output_directory = r'C:\Users\Jawahar\Documents\Interview_task\GMIndia\output'
    prompt = "Extract relevant available data from the following bank statement text, and return in JSON format."
    process_image(input_image, output_directory, prompt)

