# Video Transcript RAG — Local Ollama + Qdrant

This project is a Retrieval-Augmented Generation (RAG) pipeline that allows users to query transcribed video lectures using a local language model (via Ollama) and vector similarity search (via Qdrant). It demonstrates how to embed video subtitles, store them in a vector DB, and generate answers using Mistral or any local LLM.

---

##  Features

-  Question answering from video transcripts
-  Subtitle chunking + sentence splitting using spaCy
-  Semantic search with Qdrant
-  Local LLM inference with Ollama (`mistral`, `llama2`, etc.)
-  Topic modeling using BERTopic
-  Interactive frontend via Gradio

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ImReaper10/AIProject.git
cd AIProject
```

### Set Up Virtual Environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows
pip install -r requirements.txt

### Install & Run Qdrant Locally
Download from Qdrant Releases, extract, and run:
./qdrant.exe

### How to Run the Project
Step 1: Preprocess the subtitles
```python
python sentence_processing.py
```
Step 2: Upload vectors to Qdrant
```python
python prepare_vectors.py
```

Step 3: Start the app (with Ollama running)
```bash
ollama run mistral
```
```python
python app.py
```

### Example Query
Using only the videos, explain the advantages of CNNs over fully connected networks.
Using only the videos, explain the the binary cross entropy loss function.

