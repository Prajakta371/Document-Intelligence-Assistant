from app.embeddings import embed
from app.store import VectorStore

chunks = [
    "Employees may work from home up to three days per week with manager approval.",
    "Annual leave must be requested at least two weeks in advance.",
    "The office kitchen is cleaned every Friday afternoon.",
]

store = VectorStore()
store.add_document("policy.txt", chunks, embed(chunks))

question = "How many days can I work remotely?"
for hit in store.search(embed([question])[0], k=3):
    print(round(hit["score"], 2), hit["text"])