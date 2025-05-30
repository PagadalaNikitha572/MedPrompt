import subprocess
import json
import re

def call_ollama_llm(prompt_text, model="mistral"):
    """
    Calls the local LLM via Ollama with the given prompt and returns the output.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt_text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("LLM call failed:", result.stderr.decode())
        return ""
    return result.stdout.decode()


def clean_json_text(text: str) -> str:
    """
    Cleans common issues in JSON text from LLM:
    - Remove JS-style comments
    - Remove trailing commas before } or ]
    - Quote standalone percentages
    """
    # Remove // comments
    text = re.sub(r'//.*?\n', '\n', text)

    # Remove trailing commas in objects or arrays
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Quote unquoted percentages (e.g. 88% => "88%")
    text = re.sub(r'(?<=:\s)(\d+%)', r'"\1"', text)

    return text



def extract_combined(text, model="mistral"):
    """
    Extracts both concise and detailed structured outputs from the clinical note using one model call.
    """
    prompt = f"""
You are a clinical language assistant.

First, give a **concise structured summary** of the medical data in JSON format using only the following keys if present:
- Medication, Dose, Unit, Time, Date, Symptom, Diagnosis, Procedure, LabTest, VitalSign

Then, give a **detailed structured breakdown** of the information using custom categories based on what's actually mentioned.

Input:
\"\"\"{text}\"\"\"

Output:
{{
  "ConciseSummary": {{
    ...
  }},
  "DetailedSummary": {{
    ...
  }}
}}
    """

    response = call_ollama_llm(prompt, model=model)

    # Extract JSON substring: try from first '{' to last '}', inclusive
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        json_str = response[start:end]

        # Clean json text for common formatting issues
        json_str = clean_json_text(json_str)

        # Parse JSON
        structured = json.loads(json_str)
        return structured
    except Exception as e:
        print("❌ Failed to extract JSON:", e)
        print("🔁 Raw model output:\n", response)
        # Return raw response and error for debugging
        return {"RawResponse": response, "error": str(e)}
