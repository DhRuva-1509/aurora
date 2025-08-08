import pandas as pd
import json
import csv

# Use robust CSV reading in case of unescaped commas
df = pd.read_csv("valid.csv", quoting=csv.QUOTE_NONE, on_bad_lines='skip', encoding='utf-8')

# Replace _comma_ with actual commas
df['utterance'] = df['utterance'].astype(str).str.replace("_comma_", ",")
df['context'] = df['context'].astype(str).str.replace("_comma_", ",")

instruction_output_pairs = []

# Group by conversation
for _, group in df.groupby("conv_id"):
    group = group.sort_values("utterance_idx").reset_index(drop=True)

    for i in range(len(group) - 1):
        a = group.iloc[i]
        b = group.iloc[i + 1]

        instruction = f"Text: {a['utterance'].strip()}\nSentiment: {a['context'].strip()}"
        output = b['utterance'].strip()

        instruction_output_pairs.append({
            "instruction": instruction,
            "output": output
        })

# Save to JSONL
with open("empathetic_dialogues_advice_valid.jsonl", "w") as f:
    for ex in instruction_output_pairs:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Total instruction-output pairs: {len(instruction_output_pairs)}")
