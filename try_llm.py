from app.embeddings import embed
from app.store import VectorStore
from app.llm import answer

chunks = [
    "Employees may work from home up to three days per week with manager approval.",
    "Annual leave must be requested at least two weeks in advance.",
    "The office kitchen is cleaned every Friday afternoon.",
]

store = VectorStore()
store.add_document("policy.txt", chunks, embed(chunks))

for question in ["How many days can I work remotely?", "What is the parental leave policy?"]:
    hits = store.search(embed([question])[0], k=3)
    result = answer(question, hits)
    print("\nQ:", question)
    print("A:", result["answer"])