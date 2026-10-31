from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Load the model once and reuse it (loading is slow)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """Turn a list of strings into a (n, 384) float32 array of unit vectors."""
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")