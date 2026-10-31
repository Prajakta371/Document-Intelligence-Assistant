import uuid

import faiss
import numpy as np

DIM = 384  # size of each embedding vector from all-MiniLM-L6-v2


class VectorStore:
    """Keeps document chunks and their vectors, and finds the closest chunks to a question."""

    def __init__(self):
        # IndexFlatIP = compare vectors by dot product (= cosine similarity, because our vectors are normalised)
        self.index = faiss.IndexFlatIP(DIM)
        # records[i] describes the chunk stored at position i in the index
        self.records = []

    def add_document(self, name, chunks, vectors):
        """Store a document's chunks. Returns a short document id."""
        doc_id = uuid.uuid4().hex[:8]
        self.index.add(vectors)  # vectors: array of shape (number_of_chunks, 384)
        for i, text in enumerate(chunks):
            self.records.append(
                {"doc_id": doc_id, "doc_name": name, "chunk_no": i + 1,
                 "text": text, "vector": vectors[i]}
            )
        return doc_id

    def search(self, query_vector, k=3):
        """Return the k most similar chunks, best first."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, positions = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            record = dict(self.records[pos])
            record.pop("vector")  # callers don't need the raw vector
            record["score"] = float(score)
            results.append(record)
        return results

    def delete_document(self, doc_id):
        """Remove a document. FAISS flat indexes can't delete, so we rebuild without it."""
        kept = [r for r in self.records if r["doc_id"] != doc_id]
        if len(kept) == len(self.records):
            return False
        self.index = faiss.IndexFlatIP(DIM)
        self.records = kept
        if kept:
            self.index.add(np.array([r["vector"] for r in kept], dtype="float32"))
        return True

    def list_documents(self):
        docs = {}
        for r in self.records:
            d = docs.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "name": r["doc_name"], "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())
        