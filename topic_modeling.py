from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import json

def run_topic_model(sentences, save_path="data/topic_sentences.json"):
    embedder = SentenceTransformer("clip-ViT-B-32")
    texts = [s["text"] for s in sentences]

    topic_model = BERTopic(embedding_model=embedder, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(texts)

    # Add topics to metadata
    for i, t in enumerate(topics):
        sentences[i]["topic_id"] = t

    # Save output
    with open(save_path, "w") as f:
        json.dump(sentences, f, indent=2)

    return sentences, topic_model
