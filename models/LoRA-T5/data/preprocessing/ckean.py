import json
import re

INPUT_PATH = "clean_flan_t5_valid.jsonl"
OUTPUT_PATH = "filtered_flan_t5_valid.jsonl"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_story(instruction):
    match = re.search(r"Text:\s*(.*?)\nSentiment:", instruction, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    return clean_text(instruction)

with open(INPUT_PATH, "r") as f:
    lines = [json.loads(line) for line in f]

filtered = []
bad_lines = []

for line in lines:
    instruction = line["instruction"]
    output = line["output"]

    story = extract_story(instruction)
    cleaned_output = clean_text(output)

    # Check if output is same as story or subset
    if cleaned_output in story or story in cleaned_output:
        bad_lines.append(story)
    elif cleaned_output == "" or len(cleaned_output) < 3:
        bad_lines.append(story)
    else:
        filtered.append(line)

with open(OUTPUT_PATH, "w") as f:
    for item in filtered:
        f.write(json.dumps(item) + "\n")

print(f"✅ Filtered dataset written to {OUTPUT_PATH}")
print(f"✔️ Kept {len(filtered)} lines, 🗑️ Skipped {len(bad_lines)} bad lines.")
