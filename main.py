import os
import json
from pathlib import Path
from docx import Document
import PyPDF2
from dotenv import load_dotenv
from google import genai


def load_env():
    """Load environment variables and return Gemini API key."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env file")
    return api_key


def init_gemini():
    """Initialize and return the Gemini client."""
    api_key = load_env()
    client = genai.Client(api_key=api_key)
    return client


def extract_text(path: str) -> str:
    """Extract text content from txt, pdf, or docx files."""
    ext = path.lower().split('.')[-1]
    if ext == "txt":
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == "pdf":
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    elif ext == "docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx")


def generate_questions(client, resume: str, jd: str, n: int = 6) -> list[str]:
    """Use Gemini to generate n interview questions based on resume and JD."""
    prompt = (
        f"Given the following résumé:\n{resume}\n\n"
        f"And the following job description:\n{jd}\n\n"
        f"Your task is to generate the top {n} most relevant interview questions tailored to the candidate’s skill and experience level, and aligned with the job requirements.\n\n"
        "Instructions:\n"
        "1. Carefully analyze the résumé to determine the candidate's level (e.g., fresher, intermediate, experienced).\n"
        "2. Match the question difficulty and content appropriately.\n"
        f"3. Return only a cleanly numbered list (e.g., 1. ..., 2. ..., etc) of {n} concise, high-quality questions. No introduction or explanation."
    )

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = resp.text.strip()

    # Extract numbered questions
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line:
            parts = line.split('.', 1)
            if len(parts) > 1:
                q = parts[1].strip()
                questions.append(q)
        if len(questions) >= n:
            break

    return questions


def follow_up_question(client, history: list[dict]) -> str:
    """Generate a follow-up interview question based on past Q&A."""
    prompt = "Based on the following Q&A history, generate the next best interview question. Return only the question text."
    for i, qa in enumerate(history, start=1):
        prompt += f"\n{i}. Q: {qa['question']} A: {qa['answer']}"
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return resp.text.strip()


def score_history(client, history: list[dict]) -> str:
    """Score each Q&A pair and return structured feedback as JSON."""
    prompt = (
        "You are given a series of interview questions and answers, along with the candidate’s résumé and the job description.\n"
        "Evaluate each answer based on how well it aligns with the candidate's experience level and the expectations of the job role.\n"
        "Respond only in strict JSON format: { 'scores': [int, ...], 'feedback': str }"
    )
    prompt += "\nHistory:" + json.dumps(history, indent=2)
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return resp.text.strip()


def save_history(history: list[dict], path: str):
    """Save the Q&A history to a file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def run_qna_pipeline(resume_path: str, jd_path: str, history_path: str, flask_mode: bool = False):
    """
    Generate interview questions using Gemini and save initial Q&A history.

    Returns:
        dict: { "questions": [str, ...] }
    """
    client = init_gemini()
    resume = extract_text(resume_path)
    jd = extract_text(jd_path)

    questions = generate_questions(client, resume, jd)
    history = []

    if flask_mode:
        for q in questions:
            history.append({"question": q, "answer": "<user_input_required>"})
        save_history(history, history_path)
        return {"questions": questions}

    # If not in flask_mode, voice/CLI mode is not yet implemented
    raise NotImplementedError("Voice mode is not supported outside Flask mode.")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    print("This module is not intended to be run directly.")
