import pytest

from app.ingest import chunk_text, extract_text


# ---------- extract_text ----------

def test_extract_text_reads_txt():
    data = "Late fee is $35.".encode("utf-8")
    assert extract_text("policy.txt", data) == "Late fee is $35."


def test_extract_text_reads_markdown():
    data = "# Title".encode("utf-8")
    assert extract_text("notes.md", data) == "# Title"


def test_extract_text_ignores_file_extension_case():
    assert extract_text("POLICY.TXT", b"hello") == "hello"


def test_extract_text_skips_bad_bytes_instead_of_crashing():
    # \xff is not valid UTF-8; the code uses errors="ignore"
    assert extract_text("a.txt", b"ab\xffcd") == "abcd"


def test_extract_text_rejects_unsupported_type():
    with pytest.raises(ValueError):
        extract_text("photo.png", b"not text")


# ---------- chunk_text ----------

def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_whitespace_is_collapsed():
    assert chunk_text("a   b\n\nc\td") == ["a b c d"]


def make_long_text(n_words=300):
    return " ".join(f"w{i}" for i in range(n_words))


def test_long_text_is_split_into_several_chunks():
    chunks = chunk_text(make_long_text(), chunk_size=100, overlap=30)
    assert len(chunks) > 1


def test_no_chunk_is_longer_than_chunk_size():
    chunks = chunk_text(make_long_text(), chunk_size=100, overlap=30)
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_overlap_so_context_is_not_lost():
    chunks = chunk_text(make_long_text(), chunk_size=100, overlap=30)
    for before, after in zip(chunks, chunks[1:]):
        # the start of each chunk must already appear in the previous one
        assert after[:10] in before


def test_first_and_last_words_are_kept():
    chunks = chunk_text(make_long_text(300), chunk_size=100, overlap=30)
    assert chunks[0].startswith("w0 ")
    assert chunks[-1].endswith("w299")
    