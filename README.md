# Cyfuture_work


# 🎙️ HireIntelIQ – AI-Powered Voice Interview System

**HireIntelIQ** is an intelligent, browser-based voice interview platform that automates candidate screening using AI. With real-time voice interaction, intelligent question generation, and semantic scoring, it transforms traditional hiring into a smart, data-driven process.

---

## 🚀 Features

- 🎤 **Voice-Based Interview**: Real-time voice Q&A using Web Speech API.
- 🧠 **AI-Generated Questions**: Contextual questions tailored from resume and job description via Gemini AI.
- 🧮 **Semantic Scoring**: Cosine similarity and chunk-based scoring using SentenceTransformers.
- 📊 **Results Dashboard**: Interview performance visualization and CSV export.
- 💾 **Session Management**: Store Q&A pairs, transcripts, and scoring history.
- 🔐 **Secure Uploads**: Validates and handles PDFs, DOCX, and TXT resumes/JDs.

---

## 🧱 Tech Stack

### 🔙 Backend
- **Flask** – REST API server
- **Python** – Core logic
- **Google Gemini AI (Flash 2.0)** – Intelligent question generation
- **SentenceTransformers** – Semantic analysis (all-MiniLM-L6-v2)
- **scikit-learn** – Cosine similarity scoring
- **PyPDF2 / python-docx** – File parsing

### 🌐 Frontend
- **HTML/CSS/JS** – Responsive UI (glassmorphism + animated UX)
- **Web Speech API** – Voice recognition and synthesis

---

## 🗂️ Folder Structure

project/
├── data/
│ ├── history.json # Q&A logs
│ ├── transcript.txt # Captured voice answers
│ ├── interview_results.json # Interview records
│ ├── top.json # Top candidates
│ └── latest_files.json # Uploaded files reference
├── uploads/ # Uploaded resumes and JDs
├── templates/
│ └── index.html # Main UI template
├── app.py # Flask server and API routes
├── main.py # File processing & AI integration
├── scorer.py # Semantic scoring logic
├── evaluator.py # Legacy evaluator (alternative logic)
└── README.md



---

## 🔁 Workflow

### 1. Upload Phase
- Upload resume and job description (PDF/DOCX/TXT)
- Files validated and parsed for text

### 2. Question Generation
- Gemini AI analyzes content
- Tailors questions based on role and experience

### 3. Voice Interview
- Questions read aloud using TTS
- Candidate responses recorded via speech recognition
- Transcripts displayed live

### 4. Scoring & Analysis
- Real-time semantic scoring of Q&A
- Category-wise breakdown (Technical, Communication, etc.)
- Leaderboard and analytics in Results Dashboard

---

## 📊 Scoring Model

- **Cosine Similarity** between question and answer embeddings
- **Chunk-Based Evaluation** (10 segments)
- **Categories Evaluated**:
  - Technical Knowledge
  - Communication Skills
  - Problem Solving
  - Relevant Experience
  - Cultural Fit

---

## 🛡️ Security & Reliability

- Secure file handling using `werkzeug.utils.secure_filename`
- Max file size: 16MB
- MIME type validation
- Auto-cleanup of old sessions

---

## 📈 Results Dashboard

- View top candidates
- Sort by score, name, or date
- Export results as CSV
- Delete individual or all results

