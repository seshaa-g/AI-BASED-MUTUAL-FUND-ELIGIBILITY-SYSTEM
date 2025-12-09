import requests
import json
import pandas as pd
import fitz  # PyMuPDF
import os

# === Step 1: File Reading Functions ===

def read_file_content(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".xlsx":
        return read_excel(file_path)
    elif ext == ".json":
        return read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use .pdf, .xlsx, or .json")

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text.strip()

def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_markdown(index=False)

def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return json.dumps(json.load(f), indent=2)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return json.dumps(json.load(f), indent=2)

# === Step 2: Claude Setup ===

api_key = "YOUR_API_KEY"  # replace with your API key
file_path = r"D:\Tech enhance\Mutual_fund_eligibility\generate_bank_dataset\eligible_mutual_fund_customers.json"  # update file path
user_question = input("Enter your question based on the file: ")  # natural prompt 

# === Step 3: Build the Prompt ===

try: 
    file_content = read_file_content(file_path)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

combined_prompt = f"""
You are a helpful assistant. Use ONLY the content below to answer the question.

### File Content:
{file_content}

### Question:
{user_question}

Make sure the output is in proper JSON format if the user requests JSON.
"""

# === Step 4: Call Claude API ===

url = "https://api.anthropic.com/v1/messages"
headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

body = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 1000,
    "temperature": 0.3,
    "messages": [
        {
            "role": "user",
            "content": combined_prompt
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(body))

# === Step 5: Handle Response ===

if response.status_code == 200:
    result = response.json()
    print("\nClaude says:\n", result["content"][0]["text"])
else:
    print("Error:", response.status_code)
    print(response.text)
