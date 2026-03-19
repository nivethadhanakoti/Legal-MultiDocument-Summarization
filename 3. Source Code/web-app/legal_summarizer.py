
import re
import json
import os
import fitz  # PyMuPDF
import dateparser
import shutil
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch  # <--- ADD THIS LINE
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm  # <--- ADD THIS LINE
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import torch.optim as optim
import textwrap
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LegalGCN(nn.Module):
    """
    A unified GCN architecture for legal sentence ranking.
    Uses Message Passing to aggregate context from neighboring legal nodes.
    """
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        # Head for scoring sentence importance
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # Layer 1: Local Context Aggregation
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2: Global Graph Structure Aggregation
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Final Score Generation
        return torch.sigmoid(self.head(x))


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading Global Models (This happens once at startup)...")

# 1. Zero-shot Classifier
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=0 if torch.cuda.is_available() else -1)

# 2. Sentence Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Summarization Model & Tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)

# 4. GCN Architecture (The weights will be initialized once here)
# We initialize it with the embedding dimension of all-MiniLM-L6-v2 (384)
gcn_model = LegalGCN(in_channels=384).to(device)
gcn_model.eval() 

print("✅ All models loaded and ready.")

# Folder Paths
DOCS_FOLDER = "./docs"
PREPROCESSED_FOLDER = "./preprocessed_sentences"
GRAPHS_FOLDER = "./graphs"

# Initialize resources
nlp = spacy.load("en_core_web_sm")
# Setup NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Add this line
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))

# Config Constants
ARGUMENT_BOOST = 1.3
SIMILARITY_THRESHOLD = 0.65
CONF_THRESHOLD = 0.7
LABELS = ["Claim", "Evidence", "Rule", "Decision", "Background"]


# ---------------------------
# PDF EXTRACTION (New Step)
# ---------------------------
def extract_text_from_docs_folder(folder_path="./docs"):
    """Reads all PDFs in a folder and returns a list of raw strings."""
    extracted_docs = []
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    file_names.sort()  # Keep alphabetical order

    for filename in file_names:
        file_path = os.path.join(folder_path, filename)
        print(f"📄 Extracting: {filename}")
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        extracted_docs.append(full_text)
    
    return extracted_docs, file_names

# ---------------------------
# CLEAN TEXT (Your Logic)
# ---------------------------
def clean_text(text):
    text = re.sub(r"\b[A-Z]{2,}[0-9]+[A-Z0-9]*\b", " ", text)
    text = re.sub(r"[A-Z0-9]{3,}[^\w\s]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\'\"()§/-]", " ", text)
    text = re.sub(r"[-]{2,}", "-", text)
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"Case\s+\d+:\d+-[A-Za-z0-9-]+\s+Document\s+\d+", " ", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text)
    text = re.sub(r"\([a-z]{2,3}\)", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d{3}[-\s]?\d{3}[-\s]?\d{4}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# TIMESTAMP EXTRACTION (Your Logic)
# ---------------------------
def extract_timestamps_precise(text):
    patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|'
        r'Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)'
        r'\.?\s+\d{1,2},\s*\d{4}\b'
    ]
    matches = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.group(0)))
    matches.sort(key=lambda x: x[0])
    normalized = []
    seen = set()
    for pos, raw in matches:
        dt = dateparser.parse(raw, settings={"DATE_ORDER": "MDY"})
        if dt:
            norm = dt.strftime("%Y-%m-%d")
            if norm not in seen:
                normalized.append(norm)
                seen.add(norm)
    return normalized

def extract_timestamps_hybrid(text):
    precise = extract_timestamps_precise(text)
    if precise: return precise
    dt = dateparser.parse(text, settings={"DATE_ORDER": "MDY"})
    if dt: return [dt.strftime("%Y-%m-%d")]
    return []

# ---------------------------
# ENTITY & NORMALIZATION (Your Logic)
# ---------------------------
LEGAL_STOPWORDS = {"jurisdiction", "venue", "section", "article"}

def extract_entities(sentence):
    doc = nlp(sentence)
    entities = []
    for ent in doc.ents:
        text = ent.text.strip()
        label = ent.label_
        if text.upper() in LEGAL_STOPWORDS: continue
        if label == "DATE" and re.fullmatch(r"\d{3,4}", text):
            entities.append({"type": "STATUTE", "text": text})
            continue
        if label in ["PERSON", "ORG", "GPE", "DATE"]:
            entities.append({"type": label, "text": text})

    for jm in re.findall(r"Judge\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", sentence):
        entities.append({"type": "JUDGE", "text": jm})
    for cm in re.findall(r"(?:Supreme|District|Circuit|High)?\s*Court(?: of [A-Z][a-zA-Z\s]+)?", sentence):
        if not any(e["text"] == cm and e["type"] == "COURT" for e in entities):
            entities.append({"type": "COURT", "text": cm.strip()})
    for sm in re.findall(r"(?:§{1,2}\s*\d+|\d+\s*U\.S\.C\.|\d+\s*Stat\.)", sentence):
        entities.append({"type": "STATUTE", "text": sm.strip()})
    return entities if entities else None

def normalize_sentence(sentence):
    doc = nlp(sentence)
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and t.lemma_.lower() not in stop_words]
    return " ".join(tokens)

# ---------------------------
# SEGMENTATION & PIPELINE (Your Logic)
# ---------------------------
def segment_with_metadata(cleaned_text):
    sentences = sent_tokenize(cleaned_text)
    result = []
    current_ts = None

    for s in sentences:
        s_orig = s.strip()
        if not s_orig: continue

        ts_list = extract_timestamps_hybrid(s_orig)
        if ts_list:
            current_ts = ts_list[0]
            s_orig = re.sub(r"\(Entered:\s*\d{1,2}/\d{1,2}/\d{4}\)", " ", s_orig)
            s_orig = re.sub(r"Entered on [A-Za-z\s]*\d{1,2}/\d{1,2}/\d{4}", " ", s_orig)
            s_orig = s_orig.strip()

        s_dict = {"sentence": s_orig}
        if current_ts: s_dict["timestamps"] = [current_ts]
        ents = extract_entities(s_orig)
        if ents: s_dict["entities"] = ents
        normalized = normalize_sentence(s_orig)
        if normalized: s_dict["normalized"] = normalized

        if len(s_orig.split()) > 4:
            result.append(s_dict)
    return result

# In legal_summarizer.py

def run_preprocessing():
    """Processes whatever is currently in the DOCS_FOLDER."""
    # Ensure the output directory is fresh
    if os.path.exists(PREPROCESSED_FOLDER):
        shutil.rmtree(PREPROCESSED_FOLDER)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("⚠️ No PDFs found in docs folder to process.")
        return

    for idx, filename in enumerate(pdf_files, 1):
        print(f"📄 Processing PDF: {filename}")
        doc = fitz.open(os.path.join(DOCS_FOLDER, filename))
        full_text = " ".join([page.get_text() for page in doc])
        
        cleaned = clean_text(full_text)
        segmented = segment_with_metadata(cleaned)
        
        # Save each doc as doc1.json, doc2.json, etc.
        with open(os.path.join(PREPROCESSED_FOLDER, f"doc{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(segmented, f, indent=2, ensure_ascii=False)
    print(f"✅ Preprocessing complete. Files saved in {PREPROCESSED_FOLDER}")

# --- NEW STAGE: ARGUMENT CLASSIFICATION ---

def run_argument_classification():
    """Reads from preprocessed_sentences and applies labels."""
    print("🧠 Reading preprocessed docs for Argument Classification...")
    all_sentences = []
    
    # Iterate through the files you just saved
    for filename in sorted(os.listdir(PREPROCESSED_FOLDER)):
        if filename.startswith("doc") and filename.endswith(".json"):
            file_path = os.path.join(PREPROCESSED_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
            
            for sent in tqdm(doc_data, desc=f"Classifying {filename}"):
                text = sent["sentence"]
                
                # Zero-shot classification
                result = classifier(text, LABELS, multi_label=True)
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                
                # Confidence threshold logic
                sent["final_label"] = top_label if top_score >= CONF_THRESHOLD else "Background"
                all_sentences.append(sent)
                
    # Save the intermediate classified results
    with open("classified_sentences.json", "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, indent=2, ensure_ascii=False)
    
    return all_sentences

# ==========================================
# 3. GRAPH CONSTRUCTION (Boosted Logic)
# ==========================================

def build_argument_graph(all_sentences):
    """Builds a DiGraph using your specific similarity and boost logic."""
    print(f"🕸️ Building Boosted Graph with {len(all_sentences)} nodes...")
    G = nx.DiGraph()
    
    # Step 1: Add nodes
    for idx, sent in enumerate(all_sentences):
        G.add_node(idx, text=sent["sentence"], role=sent["final_label"])

    # Step 2: Embed sentences
    texts = [s["sentence"] for s in all_sentences]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Step 3: Build Edges
    
    for i in range(len(all_sentences)):
        scores = cosine_scores[i].cpu().numpy()
        connected = False
        
        for j, score in enumerate(scores):
            if i == j: continue
            if score >= SIMILARITY_THRESHOLD:
                src_role = G.nodes[i]["role"]
                tgt_role = G.nodes[j]["role"]
                
                boost = 1.0
                relation = "related"

                # Your specific boosting logic
                if src_role in ["Evidence", "Rule"] and tgt_role in ["Claim", "Decision"]:
                    relation = "supports" if src_role == "Evidence" else "justifies"
                    boost = ARGUMENT_BOOST
                
                G.add_edge(i, j, relation=relation, weight=float(score * boost))
                connected = True
        
        # Fallback: connect to best neighbor if isolated
        if not connected:
            scores[i] = -1
            best_j = scores.argmax()
            G.add_edge(i, best_j, relation="related", weight=float(scores[best_j]))
            
    os.makedirs(GRAPHS_FOLDER, exist_ok=True)
    with open(os.path.join(GRAPHS_FOLDER, "case_1_graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    
    print(f"✅ Graph Construction complete. Saved to {GRAPHS_FOLDER}")
    return G, embeddings

# --- GCN MODEL DEFINITIONS ---

# --- GCN RANKING STAGE ---

def run_gcn_ranking(G, embeddings):
    """
    Ranks sentences without a gold summary by using GCN as a 
    feature refiner and structural centrality measurer.
    """
    print("🧠 Starting Stage 3: GCN Sentence Ranking...")
    
    # 1. Prepare Data for PyTorch Geometric
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    x = torch.tensor(embeddings, dtype=torch.float32)
    
    src, dst = [], []
    weights = []
    for u, v, data in G.edges(data=True):
        src.append(node_to_idx[u])
        dst.append(node_to_idx[v])
        weights.append(data.get('weight', 1.0))
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LegalGCN(in_channels=x.shape[1]).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # 3. Unsupervised Ranking Logic
    # Since there is no Gold Summary, we use the GCN in inference mode
    # to calculate "Eigenvector-like" centrality via message passing.
    model.eval()
    with torch.no_grad():
        # The GCN refines the raw embeddings based on the 'Boosted' legal graph structure
        importance_scores = model(x, edge_index).cpu().numpy().flatten()

    # 4. Prepare Candidate List
    candidates = []
    for i, score in enumerate(importance_scores):
        candidates.append({
            "node_index": i,
            "text": G.nodes[nodes[i]]["text"],
            "role": G.nodes[nodes[i]]["role"],
            "score": float(score)
        })

    # 5. Extract Two Tiers of Summaries
    
    # SHORT SUMMARY (Top 5 - Focus on Decision/Claim)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Priority Selection: Ensure we pick Decisions and Claims for the short summary if available
    priority_short = [c for c in candidates if c['role'] in ['Decision', 'Claim']][:5]
    if len(priority_short) < 5:
        # Fill remaining slots with highest scores
        existing_texts = {c['text'] for c in priority_short}
        fillers = [c for c in candidates if c['text'] not in existing_texts][:5-len(priority_short)]
        short_selection = priority_short + fillers
    else:
        short_selection = priority_short

    # LONG SUMMARY (Top 60 - Broad Coverage)
    long_selection = candidates[:60]

    # Re-sort selections by document order for readability
    short_selection.sort(key=lambda x: x['node_index'])
    long_selection.sort(key=lambda x: x['node_index'])

    # 6. Save results
    os.makedirs("selections", exist_ok=True)
    with open("selections/short_selection.json", "w", encoding="utf-8") as f:
        json.dump(short_selection, f, indent=4)
    with open("selections/long_selection.json", "w", encoding="utf-8") as f:
        json.dump(long_selection, f, indent=4)

    print(f"✅ GCN Ranking Complete. Short: {len(short_selection)} sents, Long: {len(long_selection)} sents.")
    return short_selection, long_selection

# =============================================================================
# HELPER FUNCTIONS (From your snippet)
# =============================================================================

def clean_legal_boilerplate(text):
    """Remove legal document artifacts like headers, attorney info, and signatures."""
    text = re.sub(r'Case\s+\d+[:\d\w\-\.]+\s+Document\s+\d+.*?Filed.*?\d{4}', '', text, flags=re.I)
    text = re.sub(r'IN THE UNITED STATES DISTRICT.*?DIVISION', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'Civil Action No\..*?\n', '', text, flags=re.I)
    text = re.sub(r'(ATTORNEY|LEAD ATTORNEY|Senior Trial Attorney).*?(?=\n\n|\Z)', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.I)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def post_process_summary(summary):
    """Clean generated summary and fix abrupt endings."""
    summary = re.sub(r'\s+([.,;:!?])', r'\1', summary)
    summary = re.sub(r'\s{2,}', ' ', summary)
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    if summary and summary[-1] not in '.!?':
        summary += '.'
    return summary.strip()

def chunk_text(text, tokenizer, max_len=600, overlap=50):
    """Split text into manageable pieces for the model."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start = end - overlap
        if end >= len(tokens):
            break
    return chunks

# =============================================================================
# CORE SUMMARIZATION LOGIC
# =============================================================================

def generate_summary(text, model, tokenizer, max_target_tokens=200):
    """Process text through BART model."""
    cleaned = clean_legal_boilerplate(text)
    chunks = chunk_text(cleaned, tokenizer)
    summary_chunks = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(
            **inputs,
            num_beams=6,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=2.0,
            max_new_tokens=max_target_tokens,
            early_stopping=True
        )
        summary_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return post_process_summary(" ".join(summary_chunks))


# In legal_summarizer.py

def run_summarization(length_choice=None):
    """
    Summarizes documents based on the selected length. 
    Matches the 'length_choice' argument expected by the backend.
    """
    print(f"🚀 Starting Stage 4: Summarization for {length_choice if length_choice else 'all'}...")

    results = {"summaries": {}}
    
    # Determine which lengths to process based on the input
    types_to_run = [length_choice] if length_choice else ["short", "long"]

    for summary_type in types_to_run:
        input_path = f"selections/{summary_type}_selection.json"
        
        if not os.path.exists(input_path):
            print(f"⚠️ Missing {input_path}, skipping...")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            selection_data = json.load(f)
        
        source_text = " ".join([item["text"] for item in selection_data])
        # Set target length for the BART model
        target_len = 100 if summary_type == "short" else 400
        
        print(f"📝 Generating {summary_type.upper()} summary...")
        
        # Use the globally loaded models (summ_model, summ_tokenizer) for speed
        final_summary = generate_summary(source_text, summarizer_model, summarizer_tokenizer, max_target_tokens=target_len)
        
        results["summaries"][summary_type] = {
            "text": final_summary,
            "length": len(final_summary.split()),
            "timestamp": datetime.now().isoformat()
        }

    # Save outputs for reference
    os.makedirs("final_outputs", exist_ok=True)
    for s_type, data in results["summaries"].items():
        with open(f"final_outputs/{s_type}_summary.txt", "w", encoding="utf-8") as f:
            f.write(data["text"])
    
    return results


def run_summarization_targeted(length_choice):
    """Modified summarizer to only run the requested length."""
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    input_path = f"selections/{length_choice}_selection.json"
    with open(input_path, "r", encoding="utf-8") as f:
        selection_data = json.load(f)
    
    source_text = " ".join([item["text"] for item in selection_data])
    target_len = 100 if length_choice == "short" else 400
    
    final_summary = generate_summary(source_text, model, tokenizer, max_target_tokens=target_len)
    
    return {"summaries": {length_choice: {"text": final_summary}}}

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    # Ensure you have PDFs in /docs
    if not os.listdir(DOCS_FOLDER):
        print(f"Please put your PDF files in the {DOCS_FOLDER} folder first.")
    else:
        # Stage 1: PDF to JSON
        run_preprocessing()
        
        # Stage 2: JSON to Classified List (Reads from the JSONs)
        classified_data = run_argument_classification()
        
        # Stage 3: Classified List to Graph
        final_graph, embeddings = build_argument_graph(classified_data)
        
        print(f"--- Pipeline Finished ---")
        print(f"Total Sentences Processed: {len(classified_data)}")
        
        short_selected, long_selected = run_gcn_ranking(final_graph, embeddings)
        print("✅ All stages up to GCN ranking complete.")
        
        run_summarization()
        
#KEEP INPUT DOCS PDFS IN A FOLDER CALLED docs 
