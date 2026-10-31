import numpy as np

from app.store import VectorStore


def make_vectors(n):
    """n fake embeddings. Each one points in its own direction, so they are
    perfectly different from each other (similarity 0) and identical to
    themselves (similarity 1). That makes the expected search result obvious."""
    return np.eye(384, dtype="float32")[:n]


def test_add_and_list_documents():
    store = VectorStore()
    store.add_document("a.txt", ["one", "two", "three"], make_vectors(3))

    docs = store.list_documents()

    assert len(docs) == 1
    assert docs[0]["name"] == "a.txt"
    assert docs[0]["chunks"] == 3


def test_search_returns_best_match_first():
    store = VectorStore()
    vectors = make_vectors(3)
    store.add_document("a.txt", ["one", "two", "three"], vectors)

    hits = store.search(vectors[1], k=3)  # ask with chunk 2's own vector

    assert hits[0]["text"] == "two"
    assert hits[0]["chunk_no"] == 2
    assert abs(hits[0]["score"] - 1.0) < 1e-5


def test_search_on_empty_store_returns_nothing():
    store = VectorStore()
    assert store.search(make_vectors(1)[0]) == []


def test_k_larger_than_store_returns_everything():
    store = VectorStore()
    store.add_document("a.txt", ["one", "two"], make_vectors(2))

    hits = store.search(make_vectors(1)[0], k=10)

    assert len(hits) == 2


def test_search_results_do_not_include_raw_vector():
    store = VectorStore()
    store.add_document("a.txt", ["one"], make_vectors(1))

    hits = store.search(make_vectors(1)[0], k=1)

    assert "vector" not in hits[0]


def test_delete_document_removes_its_chunks():
    store = VectorStore()
    vectors = make_vectors(3)
    doc_a = store.add_document("a.txt", ["one", "two"], vectors[:2])
    store.add_document("b.txt", ["three"], vectors[2:])

    assert store.delete_document(doc_a) is True

    assert [d["name"] for d in store.list_documents()] == ["b.txt"]
    hits = store.search(vectors[0], k=3)  # vector that belonged to a.txt
    assert len(hits) == 1
    assert hits[0]["doc_name"] == "b.txt"


def test_delete_unknown_document_returns_false():
    store = VectorStore()
    assert store.delete_document("nope") is False