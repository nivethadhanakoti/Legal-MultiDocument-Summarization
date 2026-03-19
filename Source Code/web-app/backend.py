from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil
import uvicorn
import uuid  # Added for unique session IDs
import legal_summarizer as pipeline

app = FastAPI()
status_tracker = {"stage": "Idle", "progress": 0}

@app.get("/progress")
async def get_progress():
    return status_tracker

@app.post("/summarize")
async def summarize_docs(files: list[UploadFile] = File(...), length: str = Form(...)):
    global status_tracker
    # Create a unique session ID to avoid "Access Denied" on shared folders
    session_id = str(uuid.uuid4())[:8]
    session_docs = f"./docs_{session_id}"
    session_preprocessed = f"./preprocessed_{session_id}"
    
    try:
        status_tracker["progress"] = 5
        status_tracker["stage"] = "📥 Creating Session & Uploading..."
        
        # Create fresh directories for THIS specific request
        os.makedirs(session_docs, exist_ok=True)
        os.makedirs(session_preprocessed, exist_ok=True)
        
        # Override the pipeline paths dynamically
        pipeline.DOCS_FOLDER = session_docs
        pipeline.PREPROCESSED_FOLDER = session_preprocessed

        for file in files:
            file_path = os.path.join(session_docs, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 2. Pipeline Stages
        status_tracker["progress"] = 20
        status_tracker["stage"] = "🔧 Preprocessing..."
        pipeline.run_preprocessing()
        
        status_tracker["progress"] = 45
        status_tracker["stage"] = "🧠 Classifying Arguments..."
        classified_data = pipeline.run_argument_classification()
        
        status_tracker["progress"] = 70
        status_tracker["stage"] = "🕸️ Building Graph..."
        final_graph, embeddings = pipeline.build_argument_graph(classified_data)
        
        status_tracker["progress"] = 85
        status_tracker["stage"] = "📊 GCN Ranking..."
        pipeline.run_gcn_ranking(final_graph, embeddings)
        
        status_tracker["progress"] = 95
        # Inside summarize_docs in backend.py
        status_tracker["stage"] = f"📝 Generating {length} Summary..."

        # Ensure the keyword 'length_choice' matches the definition in legal_summarizer.py
        summary_results = pipeline.run_summarization(length_choice=length)

        status_tracker["progress"] = 100
        status_tracker["stage"] = "Idle"
        return {"summary": summary_results["summaries"][length]["text"]}

    except Exception as e:
        status_tracker["stage"] = "Error"
        print(f"❌ Backend Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)