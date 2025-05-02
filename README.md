# Video Transcript RAG — Local Ollama + Qdrant

This project is a Retrieval-Augmented Generation (RAG) pipeline that allows users to query transcribed video lectures using a local language model (via Ollama) and vector similarity search (via Qdrant). It demonstrates how to embed video subtitles, store them in a vector DB, and generate answers using Mistral.

---

##  Features

-  Question answering from video transcripts
-  Subtitle chunking + sentence splitting using spaCy
-  Topic modeling using BERTopic
-  Local LLM inference with Ollama (`mistral`)
-  Semantic search with Qdrant
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

## Demo Videos

Below are example queries run on the video RAG system with embedded video transcripts:

- **Binary Classification Question**
  (https://img.youtube.com/vi/XcWSUFef5D8/0.jpg)](https://youtu.be/XcWSUFef5D8)
  > Querying "What is binary classification?" and answering from transcript chunks.

- **CNNs (Convolutional Neural Networks) Question**
  [![CNNs](https://img.youtube.com/vi/imd_5FKqAsU/0.jpg)](https://youtu.be/imd_5FKqAsU)
  > Demonstrates contextual answer retrieval for a CNN-related question.

- **ResNets Explanation**
  [![ResNets](https://img.youtube.com/vi/wSeNgWnlaSI/0.jpg)](https://youtu.be/wSeNgWnlaSI)
  > Tests the model's ability to recall ResNet concepts from lecture transcripts.

- **Out-of-Scope Question**
  [![Out-of-scope](https://img.youtube.com/vi/x9TNC9w-3fE/0.jpg)](https://youtu.be/x9TNC9w-3fE)
  > Asks an unrelated question to verify if the system properly declines or misfires.


