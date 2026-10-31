from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.embeddings import embed
from app.ingest import chunk_text, extract_text
from app.llm import answer
from app.store import VectorStore

app = FastAPI(title="Document Intelligence Assistant")

store = VectorStore()  # one store, held in memory while the server runs
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class AskRequest(BaseModel):
    question: str
    k: int = 3  # how many chunks to retrieve


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.list_documents())}


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB.")
    try:
        text = extract_text(file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in this file.")
    for existing in store.list_documents():
        if existing["name"] == file.filename:
            store.delete_document(existing["doc_id"])
    doc_id = store.add_document(file.filename, chunks, embed(chunks))
    return {"doc_id": doc_id, "name": file.filename, "chunks": len(chunks)}


@app.get("/documents")
def list_documents():
    return store.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    hits = store.search(embed([question])[0], k=req.k)
    try:
        return answer(question, hits)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"The AI service failed: {type(e).__name__}")


@app.get("/")
def home():
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "Document Intelligence Assistant API. Open /docs to try it."}
