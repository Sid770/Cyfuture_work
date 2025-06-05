import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts):
    return [np.array(emb) for emb in model.encode(texts)]


def evaluate_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Split into Q/A pairs
    qas = [block.strip() for block in raw.strip().split("\n\n") if block.strip()]
    if not qas:
        raise ValueError("Transcript is empty or malformed.")

    total = len(qas)

    # Split into 10 equal chunks
    chunks = np.array_split(qas, 10)

    scores = []

    for idx, chunk in enumerate(chunks):
        try:
            texts = []
            for qa in chunk:
                lines = qa.split('\n')
                q = lines[0].replace("Q: ", "").strip()
                a = lines[1].replace("A: ", "").strip() if len(lines) > 1 else ""
                texts.extend([q, a])

            embeddings = get_embeddings(texts)
            sim_scores = []

            # Compare each Q and A pair
            for i in range(0, len(embeddings) - 1, 2):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0][0]
                sim_scores.append(sim)

            avg_score = np.mean(sim_scores) if sim_scores else 0

            scores.append({
                "progress": f"{(idx + 1) * 10}%",
                "score": round(float(avg_score * 100), 2),
                "samples": len(sim_scores)
            })

        except Exception as e:
            scores.append({
                "progress": f"{(idx + 1)}",
                "score": None,
                "error": str(e)
            })

    return scores
