from io import BytesIO
from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of a PDF, .txt or .md file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at a space."""
    text = " ".join(text.split())  # collapse all whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + chunk_size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        new_start = max(end - overlap, start + 1)
        space = text.find(" ", new_start, end)
        start = space + 1 if space != -1 else new_start
    return [c for c in chunks if c]