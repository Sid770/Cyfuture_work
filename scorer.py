from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load once at the top
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_similarity_scores(transcript_path):
    """
    Reads Q&A pairs from transcript.txt and evaluates cosine similarity.
    Returns chunk-based progress scores and overall average score.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        qas = []
        question = ""
        for line in lines:
            if line.startswith("Q: "):
                question = line[3:].strip()
            elif line.startswith("A: "):
                answer = line[3:].strip()
                if question:
                    qas.append((question, answer))
                    question = ""

        if not qas:
            raise ValueError("No valid Q&A pairs found in transcript.")

        # Split into 10 chunks for progressive analysis
        total_pairs = len(qas)
        chunk_size = max(1, total_pairs // 10)
        chunks = [qas[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
        if len(chunks) > 10:
            chunks = chunks[:10]

        results = []
        all_scores = []

        for i, chunk in enumerate(chunks):
            sim_scores = []
            for question, answer in chunk:
                embeddings = model.encode([question, answer])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                sim_scores.append(score)
                all_scores.append(score)

            avg_score = np.mean(sim_scores) if sim_scores else 0.0
            results.append({
                "progress": f"{(i + 1) * 10}%",
                "score": round(avg_score * 100, 2)
            })

        overall = round(np.mean(all_scores) * 100, 2) if all_scores else 0.0

        return {
            "scores": results,
            "overall": overall
        }

    except Exception as e:
        return {
            "error": str(e),
            "scores": [],
            "overall": None
        }


def evaluate_qa_pairs(qa_pairs):
    """
    Takes a list of Q&A dicts and returns an average cosine similarity score.
    Used to score an individual candidate's interview in /submit-interview.
    """
    try:
        sim_scores = []
        for pair in qa_pairs:
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            if question and answer:
                embeddings = model.encode([question, answer])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                sim_scores.append(score)

        overall = round(np.mean(sim_scores) * 100, 2) if sim_scores else 0.0
        return overall

    except Exception as e:
        print(f"Error scoring QA pairs: {e}")
        return 0.0
