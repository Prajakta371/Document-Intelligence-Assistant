import logging
import time

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

MODEL = "claude-sonnet-5-5"  # change this one line to switch models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("llm")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered excerpts provided by the user. "
    "Cite the excerpts you used like [1] or [2] right after the sentences they support. "
    "If the excerpts do not contain the answer, say you could not find it in the documents. "
    "Do not use outside knowledge and do not guess."
)


def build_context(hits):
    """Turn retrieved chunks into a numbered list Claude can cite."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] (from {hit['doc_name']}, chunk {hit['chunk_no']})\n{hit['text']}")
    return "\n\n".join(parts)


def answer(question, hits):
    """Ask Claude to answer the question from the retrieved chunks."""
    if not hits:
        return {"answer": "No documents have been uploaded yet.", "sources": []}

    client = Anthropic()  # finds the key in the environment automatically
    prompt = f"Excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"

    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.time() - start, 2)

    log.info(
        "model=%s input_tokens=%d output_tokens=%d latency=%ss",
        MODEL, response.usage.input_tokens, response.usage.output_tokens, latency,
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {"n": i, "doc_name": h["doc_name"], "chunk_no": h["chunk_no"], "text": h["text"]}
            for i, h in enumerate(hits, start=1)
        ],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_s": latency,
    }