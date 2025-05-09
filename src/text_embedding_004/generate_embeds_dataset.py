import csv
import json
import google.genai as genai
from google.genai import types
import time
from pathlib import Path

GOOGLE_GENAI_API_KEY = "YOUR_API_KEY"
EMBEDDING_MODEL = "text-embedding-004"
OUTPUT_DIMENSIONALITY = 768
OUTPUT_FILE = f"YT-comments-{EMBEDDING_MODEL}-{OUTPUT_DIMENSIONALITY}.jsonl"
SLEEP_LENGTH = 0.3

genai = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

skip_lines = 0
output_file = Path(OUTPUT_FILE)
if output_file.exists():
    with open(output_file, "r") as file:
        skip_lines = sum(1 for _ in file)

print(skip_lines)
with open("data/train.csv", "r", encoding="utf-8") as dataset_file:
    dataset_csv = csv.DictReader(dataset_file)

    with open(output_file, "a") as output_file:
        record_count = 0
        for entry in dataset_csv:
            record_count += 1
            print(record_count)
            if (record_count <= skip_lines):
                continue

            embedding_response = genai.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=entry['comment_text'],                
                config=types.EmbedContentConfig(task_type="CLASSIFICATION")
            )

            result = {
                'id': entry['id'],
                'embeddings': embedding_response.embeddings[0].values,
                'toxic': entry['toxic'],
                'severe_toxic': entry['severe_toxic'],
                'obscene': entry['obscene'],
                'threat': entry['threat'],
                'insult': entry['insult'],
                'identity_hate': entry['identity_hate'],                                                                                
            }
            json_result = json.dumps(result)
            output_file.write(json_result + "\n")
            output_file.flush()
            time.sleep(SLEEP_LENGTH)

              
