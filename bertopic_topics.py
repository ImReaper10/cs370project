from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import json
import os

# === Step 1: Load processed sentences ===
input_file = "final_sentences.json"
output_file = "sentences_with_topics.json"

if not os.path.exists(input_file):
    raise FileNotFoundError("Run sentence_processing.py first to generate final_sentences.json")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
embedder = SentenceTransformer("clip-ViT-B-32")

# === Step 2: Apply BERTopic ===
topic_model = BERTopic(embedding_model=embedder, calculate_probabilities=True)
topics, probs = topic_model.fit_transform(texts)

# === Step 3: Add topic IDs to data ===
for i, topic_id in enumerate(topics):
    data[i]["topic_id"] = topic_id

# === Step 4: Save updated data ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved topic-labeled data to {output_file}")

# === Optional: Show bar chart ===
topic_model.visualize_barchart(top_n_topics=10).show()
