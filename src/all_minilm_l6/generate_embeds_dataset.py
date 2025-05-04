import csv
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

OUTPUT_FILE = f"SP-all-MiniLM-L6-v2.jsonl"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # vec is 384

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
            if (record_count % 10 == 0):
                print(record_count)
                
            if (record_count <= skip_lines):
                continue

            embeddings = model.encode([entry['comment_text']])

            result = {
                'id': entry['id'],
                'embeddings': embeddings[0].tolist(),
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
            

              