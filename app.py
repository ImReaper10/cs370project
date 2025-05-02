from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import gradio as gr
import requests
import json

# === Step 1: Load the embedder ===
clip_text_embedder = SentenceTransformer("clip-ViT-B-32")

# === Step 2: Connect to local Qdrant ===
client = QdrantClient(host="localhost", port=6333)
collection_name = "video_rag"

# === Step 3: Search context from Qdrant using new API ===
def search_context(query, top_k=5):
    query_vec = clip_text_embedder.encode(query).tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    return [r.payload.get("text", "") for r in results if "text" in r.payload]


# === Step 4: Stream answer from Ollama ===
def stream_answer(query):
    context_chunks = search_context(query)

    if not context_chunks:
        yield "⚠️ No relevant context found in Qdrant."
        return

    context = "\n".join(context_chunks)

    print("======== PROMPT SENT TO OLLAMA ========")
    print(context)
    print("======== END PROMPT ========")

    prompt = f"""You are a helpful AI assistant.

Answer the user's question using only the context provided below.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "mistral",
                "stream": True,
                "messages": [
                    { "role": "system", "content": "You are a retrieval-based assistant. You must only answer using the provided context. Do not use prior knowledge. If context is insufficient, say 'Context not found'." }
,
                    {"role": "user", "content": prompt}
                ]
            },
            stream=True
        )

        accumulated = ""
        for chunk in response.iter_lines():
            if not chunk or chunk.strip() == b'data: [DONE]':
                continue
            if chunk.startswith(b'data: '):
                try:
                    data = json.loads(chunk.removeprefix(b"data: ").decode("utf-8"))
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated += content
                        yield accumulated
                except Exception as e:
                    print("⚠️ Error parsing chunk:", e)
                    continue

    except Exception as e:
        print("❌ Error contacting Ollama:", e)
        yield f"❌ Ollama error: {e}"

# === Step 5: Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## Video Transcript RAG — Local Ollama + Qdrant")
    inp = gr.Textbox(label="Ask a question based on the video transcripts")
    out = gr.Textbox(label="Answer", lines=10)
    btn = gr.Button("Get Answer")
    btn.click(fn=stream_answer, inputs=inp, outputs=out)

demo.launch(debug=True)