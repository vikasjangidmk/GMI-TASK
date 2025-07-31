import json
from openai import OpenAI

def handle_json(json_text):
	json_text = json_text[json_text.find('{'):json_text.rfind('}')+1]
	return json_text

def parse_with_gpt (input_text: str, max_tokens: int = 5000) -> dict:
	"""
		This function utilizes the gpt model to parse the input text into a JSON format
	"""
	client = OpenAI()
	results = client.chat.completions.create(
		model="gpt-4o", 
		messages=[{
			"role": "system",
			"content": "Extract the data in text into JSON format",
			"role": "user", 
			"content": input_text
		}],
		temperature=0.0,
	)
	data = results.choices[0].message.content
	data = json.loads(handle_json(data))
	return data