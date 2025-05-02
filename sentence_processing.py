import os, json, spacy
from glob import glob
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

def time_difference(start, end):
    time_format = "%H:%M:%S.%f"
    start_dt = datetime.strptime(start, time_format)
    end_dt = datetime.strptime(end, time_format)
    return (end_dt - start_dt).total_seconds()

def load_and_split(folder):
    all_sentences = []
    for file in glob(f"{folder}/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            video_id = data.get("video_id", os.path.basename(file).split(".")[0])

            # Step 1: Filter short captions (<1 sec)
            captions = [
                cap for cap in data["captions"]
                if time_difference(cap["start"], cap["end"]) >= 1.0
            ]

            # Step 2: Drop every other one to reduce redundancy
            captions = [cap for idx, cap in enumerate(captions) if idx % 2 == 0]

            for c in captions:
                raw = c["text"].strip()
                ts = c["start"]
                if raw:
                    doc = nlp(raw)
                    for sent in doc.sents:
                        all_sentences.append({
                            "text": sent.text.strip(),
                            "timestamp": ts,
                            "video": video_id
                        })
    return all_sentences

if __name__ == "__main__":
    sentences = load_and_split("ProjectCaptions")
    with open("final_sentences.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sentences)} sentences to final_sentences.json")
