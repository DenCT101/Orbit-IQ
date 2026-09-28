"""
rag_ingest.py — Ingest knowledge base documents into ChromaDB.

Reads all .txt files from the knowledge_base/ directory, chunks them,
generates embeddings using sentence-transformers, and stores them in
a persistent ChromaDB collection.
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions


# ── Configuration ─────────────────────────────────────────────────────
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "plant_disease_kb"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Mapping: CNN class index → knowledge base filename ────────────────
CLASS_TO_FILE = {
    0:  "apple_scab.txt",
    1:  "apple_black_rot.txt",
    2:  "apple_cedar_rust.txt",
    3:  "apple_healthy.txt",
    4:  "background_without_leaves.txt",
    5:  "blueberry_healthy.txt",
    6:  "cherry_powdery_mildew.txt",
    7:  "cherry_healthy.txt",
    8:  "corn_cercospora_gray_leaf_spot.txt",
    9:  "corn_common_rust.txt",
    10: "corn_northern_leaf_blight.txt",
    11: "corn_healthy.txt",
    12: "grape_black_rot.txt",
    13: "grape_esca_black_measles.txt",
    14: "grape_leaf_blight.txt",
    15: "grape_healthy.txt",
    16: "orange_huanglongbing.txt",
    17: "peach_bacterial_spot.txt",
    18: "peach_healthy.txt",
    19: "pepper_bacterial_spot.txt",
    20: "pepper_healthy.txt",
    21: "potato_early_blight.txt",
    22: "potato_late_blight.txt",
    23: "potato_healthy.txt",
    24: "raspberry_healthy.txt",
    25: "soybean_healthy.txt",
    26: "squash_powdery_mildew.txt",
    27: "strawberry_leaf_scorch.txt",
    28: "strawberry_healthy.txt",
    29: "tomato_bacterial_spot.txt",
    30: "tomato_early_blight.txt",
    31: "tomato_late_blight.txt",
    32: "tomato_leaf_mold.txt",
    33: "tomato_septoria_leaf_spot.txt",
    34: "tomato_spider_mites.txt",
    35: "tomato_target_spot.txt",
    36: "tomato_yellow_leaf_curl.txt",
    37: "tomato_mosaic_virus.txt",
    38: "tomato_healthy.txt",
}


def extract_source_info(text: str) -> dict:
    """Extract SOURCE and URL metadata from the text header."""
    source_match = re.search(r"SOURCE:\s*(.+)", text)
    url_match = re.search(r"URL:\s*(\S+)", text)
    return {
        "source": source_match.group(1).strip() if source_match else "Unknown",
        "url": url_match.group(1).strip() if url_match else "",
    }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, breaking at paragraph or sentence
    boundaries when possible.
    """
    # Split into paragraphs first
    paragraphs = re.split(r"\n\s*\n", text.strip())
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph exceeds chunk_size, save current and start new
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of the current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""
        
        current_chunk += ("\n\n" if current_chunk else "") + para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle case where a single paragraph is longer than chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:  # Allow some flexibility
            final_chunks.append(chunk)
        else:
            # Split long chunks at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            sub_chunk = ""
            for sentence in sentences:
                if len(sub_chunk) + len(sentence) + 1 > chunk_size and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sentence
                else:
                    sub_chunk += (" " if sub_chunk else "") + sentence
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())
    
    return final_chunks


def ingest_knowledge_base():
    """
    Read all knowledge base documents, chunk them, and store in ChromaDB
    with metadata (source, URL, disease class index, filename).
    """
    print("=" * 60)
    print("  Orbit-IQ RAG — Knowledge Base Ingestion")
    print("=" * 60)
    
    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Use sentence-transformers for embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Delete existing collection if it exists (fresh ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    all_docs = []
    all_metadatas = []
    all_ids = []
    
    total_chunks = 0
    
    for class_idx, filename in CLASS_TO_FILE.items():
        filepath = os.path.join(KB_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ Missing: {filename} (class {class_idx})")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Extract source info
        source_info = extract_source_info(text)
        
        # Chunk the document
        chunks = chunk_text(text)
        
        print(f"  ✓ {filename}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            doc_id = f"class{class_idx}_{filename.replace('.txt', '')}_chunk{i}"
            
            all_docs.append(chunk)
            all_metadatas.append({
                "class_index": class_idx,
                "filename": filename,
                "source": source_info["source"],
                "url": source_info["url"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            all_ids.append(doc_id)
            total_chunks += 1
    
    # Batch insert into ChromaDB
    BATCH_SIZE = 100
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(all_docs))
        collection.add(
            documents=all_docs[i:batch_end],
            metadatas=all_metadatas[i:batch_end],
            ids=all_ids[i:batch_end],
        )
    
    print(f"\n{'=' * 60}")
    print(f"  Ingestion complete!")
    print(f"  Total documents: {len(CLASS_TO_FILE)}")
    print(f"  Total chunks stored: {total_chunks}")
    print(f"  ChromaDB path: {CHROMA_PERSIST_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"{'=' * 60}")
    
    return collection


if __name__ == "__main__":
    ingest_knowledge_base()
