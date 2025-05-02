import os
import json
import glob
import spacy
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# === Step 1: Load and combine all .json files ===
json_dir = "ProjectCaptions"
all_sentences = []

for path in glob.glob(os.path.join(json_dir, "*.json")):
    with open(path, "r") as f:
        data = json.load(f)
        video_id = data.get("video_id", os.path.basename(path).split(".")[0])
        for item in data.get("captions", []):
            sentence = item.get("text", "").strip()
            timestamp = item.get("start", 0.0)
            if sentence:
                all_sentences.append({
                    "text": sentence,
                    "video": video_id,
                    "timestamp": timestamp
                })

print(f"Loaded {len(all_sentences)} raw caption snippets.")

# === Step 2: Merge short sentences using spaCy ===
nlp = spacy.load("en_core_web_sm")
merged_sentences = []
buffer = ""
buffer_meta = []

for s in all_sentences:
    buffer += " " + s["text"]
    buffer_meta.append(s)
    if len(buffer.split()) >= 15:  # or use doc.sents
        doc = nlp(buffer.strip())
        for sent in doc.sents:
            merged_sentences.append({
                "text": sent.text.strip(),
                "video": buffer_meta[0]["video"],
                "timestamp": buffer_meta[0]["timestamp"]
            })
        buffer = ""
        buffer_meta = []

print(f"Merged into {len(merged_sentences)} longer sentences.")

# === Step 3: Generate embeddings ===
embedder = SentenceTransformer("clip-ViT-B-32")
texts = [s["text"] for s in merged_sentences]
vectors = embedder.encode(texts, convert_to_numpy=True)

# === Step 4: Connect and upload to Qdrant ===
client = QdrantClient("http://localhost:6333")
collection_name = "video_rag"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
)

points = [
    PointStruct(id=i, vector=vectors[i].tolist(), payload=merged_sentences[i])
    for i in range(len(vectors))
]


batch_size = 1000
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(collection_name=collection_name, points=batch)
    print(f"Uploaded batch {i//batch_size + 1} of {(len(points) - 1)//batch_size + 1}")

