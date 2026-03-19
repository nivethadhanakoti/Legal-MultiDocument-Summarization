# Legal Multi-Document Summarizer

## A. Dataset — URL / Source

### 1. Multi-LexSum Dataset (Primary)
**URL:** https://multilexsum.github.io/

Multi-LexSum is a dataset of 9,280 legal case summaries designed to support research on summarization of civil rights litigation. It provides summaries at multiple levels of granularity:
- One-sentence "extreme" summaries
- Paragraph-length summaries
- Long multi-paragraph summaries (500+ words)

Each case includes multiple source documents: complaints, court opinions, orders, settlement agreements, etc. Summaries are expert-authored following detailed guidelines, making this a reliable benchmark for long document summarization, multi-document summarization, and controllable summarization.

### 2. Supreme Court of India Dataset (Secondary / In Progress)
**URL:** https://www.sci.gov.in/case-status-case-no/

To validate the model on Indian legal documents, documents are collected from the Supreme Court of India website. Each case consists of multiple documents. Gold summaries for these cases are being produced in collaboration with law domain professionals.


## B. Software and Hardware Requirements

### Hardware Requirements

**Training Phase (resource-intensive):**

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7 (8th Gen or higher) or equivalent AMD |
| GPU | NVIDIA Tesla T4 or equivalent (minimum 15 GB VRAM) |
| RAM | 12 GB minimum |
| Storage | 100 GB disk space |

> The GPU is crucial for acceleration of graph construction, LLM pre-training, and summarization. RAM is required for handling large in-memory graph matrices and document embeddings.

**Execution / Inference Phase:** A modern CPU is sufficient for generating summaries after training.

### Software Requirements

- **Python:** 3.9 or higher
- **OS:** Windows / macOS / Linux

| Category | Library / Framework |
|----------|-------------------|
| NLP Framework | PyTorch 2.1.2, TensorFlow |
| NLP Libraries | Hugging Face Transformers 4.40.0 |
| NLP Libraries | sentence-transformers 2.7.0 |
| NLP Libraries | spaCy 3.x, NLTK |
| Graph Processing | NetworkX, torch-geometric |
| Data Handling | NumPy 1.26.4, Pandas |
| PDF Processing | PyMuPDF (fitz) |
| Web Framework | FastAPI, Uvicorn, Streamlit |
| Utilities | tqdm, scikit-learn, dateparser, evaluate, rouge-score, bert-score, python-multipart, requests |

> All specific library versions are pinned in `requirements.txt` to ensure reproducibility. Do not upgrade versions without testing compatibility.

---

## C. Instructions to Execute the Source Code

### Prerequisites
- Python 3.9+ installed on your system
- The source code folder should contain:
```
web-app/
├── app.py
├── backend.py
├── legal_summarizer.py
└── requirements.txt
```

### Step-by-Step Execution

**Step 1 — Navigate to the web-app directory**
```bash
cd path/to/web-app
```

**Step 2 — Create a Python virtual environment**
```bash
python -m venv venv
```

**Step 3 — Activate the virtual environment**

On Windows:
```bash
venv\Scripts\activate
```
On macOS / Linux:
```bash
source venv/bin/activate
```
You should see `(venv)` appear in your terminal prompt.

**Step 4 — Install all dependencies**
```bash
pip install -r requirements.txt
```
This may take several minutes as it downloads PyTorch, Transformers, and other large libraries.

**Step 5 — Download the spaCy English language model**
```bash
python -m spacy download en_core_web_sm
```

**Step 6 — Start the backend server**
```bash
python3 backend.py
```
Wait until you see the following message before proceeding:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```
> The first run will download AI model weights (BART, SentenceTransformer). This may take a few minutes. Keep this terminal open.

**Step 7 — Start the frontend (in a NEW terminal window)**

Open a second terminal, navigate to the web-app folder, activate the venv (Steps 1 & 3), then run:
```bash
streamlit run app.py
```
The app will open in your browser at **http://localhost:8501**

> Both terminals must remain open while using the application.

---

### How to Use the Application
1. Upload one or more legal PDF documents using the sidebar
2. Select the desired summary length: **short** or **long**
3. Click **Generate Summary** and wait for the AI pipeline to complete
4. The generated summary will appear in the main panel

---

### Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'spacy'` | `pip install spacy` |
| `Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| `Form data requires python-multipart` | `pip install python-multipart` |
| `Failed to connect to backend` in Streamlit UI | Ensure `backend.py` is running and shows the Uvicorn message before using the frontend |
| Bus error / crash on macOS during classification | Set the environment variables in the macOS section above and ensure `torch==2.1.2` is installed |
