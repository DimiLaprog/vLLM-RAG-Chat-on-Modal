# rag_ingest.py
import modal
from pathlib import Path


import json
from typing import List, Dict


# ---------- Config ----------

# Local docs directory (relative to this file)
LOCAL_DOCS_DIR = Path(__file__).parent / "docs"

# Remote path inside Modal container where docs will be mounted
REMOTE_DOCS_DIR = "/docs"

# Modal volume to store index + metadata
INDEX_VOLUME_DIR = "/index"
INDEX_VOLUME_NAME = "rag-index"
INDEX_FAISS_PATH = "/index/index.faiss"
META_JSON_PATH = "/index/meta.json"

# Embedding model name (we'll use OpenAI text-embedding-3-small)
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------- Modal image & app ----------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai>=1.40.0",
        "faiss-cpu",
        "langchain-text-splitters>=0.0.1",
        "numpy",
    )
    .add_local_dir(  # Mount local ./docs into /docs in the container
        local_path=LOCAL_DOCS_DIR,
        remote_path=REMOTE_DOCS_DIR,
    )

)

# heavy / container-only libs (optional pattern)
with image.imports():
    import numpy as np
    import faiss
    from openai import OpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create / get a volume to store index and metadata
index_volume = modal.Volume.from_name(INDEX_VOLUME_NAME, create_if_missing=True)

app = modal.App("rag-ingest",
                image=image,
                secrets=[modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"])]
                )


def _load_text_files(base_path: Path) -> List[Dict]:
    """Load all .txt and .md files under base_path into memory."""
    docs = []
    for path in base_path.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        docs.append(
            {
                "path": str(path.relative_to(base_path)),
                "text": text,
            }
        )
    return docs


def _chunk_document(doc: Dict) -> List[Dict]:
    """Split one document into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(doc["text"])
    results = []
    for idx, chunk in enumerate(chunks):
        results.append(
            {
                "doc_path": doc["path"],
                "chunk_id": idx,
                "text": chunk,
            }
        )
    return results


def _embed_chunks(client, chunks: List[Dict]) -> np.ndarray:
    """Call OpenAI embeddings API on the list of chunk texts and return a 2D numpy array."""
    texts = [c["text"] for c in chunks]

    # You can batch these if you like; for now, one big call is fine up to token limits
    # For more control, implement batching here.
    print(f"Embedding {len(texts)} chunks...")
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")


@app.function(
    volumes={INDEX_VOLUME_DIR: index_volume},
    timeout=60 * 30,  # up to 30 minutes for big corpora
)
def build_index():
    """
    Ingest all text files under /docs, chunk them, embed them,
    build a FAISS index, and store it + metadata in the rag-index Volume.
    """
    # 1. Initialize OpenAI client
    client = OpenAI()

    # 2. Load docs from mounted /docs
    docs_dir = Path(REMOTE_DOCS_DIR)
    if not docs_dir.exists():
        raise RuntimeError(f"Docs directory {docs_dir} not found in container")

    docs = _load_text_files(docs_dir)
    print(f"Found {len(docs)} document(s) under {docs_dir}")

    # 3. Chunk all documents
    all_chunks: List[Dict] = []
    for doc in docs:
        doc_chunks = _chunk_document(doc)
        all_chunks.extend(doc_chunks)

    if not all_chunks:
        raise RuntimeError("No chunks created; check input docs / chunking config")

    print(f"Created {len(all_chunks)} total chunks")

    # 4. Embed all chunks
    embeddings = _embed_chunks(client, all_chunks)
    num_vectors, dim = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape}")

    # 5. Build FAISS index (inner product / cosine-style)
    index = faiss.IndexFlatIP(dim)  # for cosine, you'd L2-normalize embeddings first
    # Optional: normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 6. Persist index and metadata to Volume at /index
    # Save FAISS index
    faiss.write_index(index, INDEX_FAISS_PATH)
    print(f"Wrote FAISS index to {INDEX_FAISS_PATH}")

    # Save metadata
    meta = {
        "chunks": all_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_vectors": num_vectors,
        "dim": dim,
    }
    with open(META_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote metadata to {META_JSON_PATH}")
    print("Index build complete.")

@app.function(
    volumes={INDEX_VOLUME_DIR: index_volume},
    timeout=60 * 5,
)
def test_query(query: str = "test query"):
    """Embed a query, search FAISS, and print the top matches."""
    import faiss
    import numpy as np
    from openai import OpenAI

    # 1. Load index + metadata from the Volume
    index = faiss.read_index(INDEX_FAISS_PATH)

    with open(META_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    print(f"Loaded index with {meta['num_vectors']} vectors, dim={meta['dim']}")

    # 2. Embed the query
    client = OpenAI()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # Normalize and search top-3
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k=3)

    print(f"\nTop matches for: {query!r}")
    for score, idx in zip(D[0], I[0]):
        chunk = chunks[int(idx)]
        preview = chunk["text"].replace("\n", " ")[:200]
        print(f"- score={score:.4f} | {chunk['doc_path']}#{chunk['chunk_id']}: {preview}...")
    
@app.local_entrypoint()
def main():
    # this runs locally and just triggers the remote job
    build_index.remote()
