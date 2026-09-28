"""
rag_evaluate.py — Evaluate the RAG retrieval pipeline.

Contains 20 test questions mapped to expected disease classes.
For each question, we check whether the retrieved passages come from
the correct knowledge base document (i.e., correct disease class).
Records and prints the hit rate.
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "plant_disease_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5   # Retrieve top-5 passages
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_evaluation_results.json")


# ── 20 Test Questions ─────────────────────────────────────────────────
# Each question is a dict with:
#   - "question": the natural language query
#   - "expected_class": the CNN class index the answer should come from
#   - "topic": brief description for logging
TEST_QUESTIONS = [
    {
        "question": "What fungus causes apple scab and how does it spread?",
        "expected_class": 0,
        "topic": "Apple Scab – causal agent"
    },
    {
        "question": "How do I treat black rot on my apple tree fruit?",
        "expected_class": 1,
        "topic": "Apple Black Rot – treatment"
    },
    {
        "question": "What are the orange spots on my apple leaves caused by cedar trees nearby?",
        "expected_class": 2,
        "topic": "Cedar Apple Rust – symptoms"
    },
    {
        "question": "What is the white powdery coating on my cherry tree leaves?",
        "expected_class": 6,
        "topic": "Cherry Powdery Mildew – identification"
    },
    {
        "question": "How do I manage gray leaf spot in my corn field?",
        "expected_class": 8,
        "topic": "Corn Gray Leaf Spot – management"
    },
    {
        "question": "What causes rust pustules on corn leaves and how to control it?",
        "expected_class": 9,
        "topic": "Corn Common Rust – control"
    },
    {
        "question": "What fungicide should I use for northern leaf blight in corn?",
        "expected_class": 10,
        "topic": "Corn Northern Leaf Blight – fungicide"
    },
    {
        "question": "How do I prevent black rot from destroying my grape crop?",
        "expected_class": 12,
        "topic": "Grape Black Rot – prevention"
    },
    {
        "question": "What is esca disease in grapes and can it be cured?",
        "expected_class": 13,
        "topic": "Grape Esca – cure"
    },
    {
        "question": "My orange tree leaves have yellow mottling and the fruit stays green. What disease is this?",
        "expected_class": 16,
        "topic": "Citrus Greening – diagnosis"
    },
    {
        "question": "What causes shot-hole symptoms on peach tree leaves?",
        "expected_class": 17,
        "topic": "Peach Bacterial Spot – symptoms"
    },
    {
        "question": "How do I manage bacterial spot on my bell pepper plants?",
        "expected_class": 19,
        "topic": "Pepper Bacterial Spot – management"
    },
    {
        "question": "What is the target-board pattern on my potato leaves?",
        "expected_class": 21,
        "topic": "Potato Early Blight – identification"
    },
    {
        "question": "How did the Irish Potato Famine relate to late blight disease?",
        "expected_class": 22,
        "topic": "Potato Late Blight – history"
    },
    {
        "question": "What causes powdery white coating on squash and pumpkin leaves?",
        "expected_class": 26,
        "topic": "Squash Powdery Mildew – cause"
    },
    {
        "question": "Why are my strawberry leaves turning dark purple and scorched at the edges?",
        "expected_class": 27,
        "topic": "Strawberry Leaf Scorch – symptoms"
    },
    {
        "question": "What is the best treatment for early blight on tomatoes?",
        "expected_class": 30,
        "topic": "Tomato Early Blight – treatment"
    },
    {
        "question": "How do spider mites damage tomato plants and how to control them?",
        "expected_class": 34,
        "topic": "Tomato Spider Mites – damage & control"
    },
    {
        "question": "What causes tomato yellow leaf curl virus and how is it transmitted?",
        "expected_class": 36,
        "topic": "TYLCV – transmission"
    },
    {
        "question": "How is tomato mosaic virus spread through mechanical contact?",
        "expected_class": 37,
        "topic": "ToMV – mechanical spread"
    },
]


def evaluate_retrieval():
    """
    Run all 20 test questions against ChromaDB and check if the
    retrieved passages belong to the expected disease class.
    
    A "hit" is counted if at least one of the top-K retrieved passages
    has a class_index matching the expected class.
    """
    print("=" * 70)
    print("  Orbit-IQ RAG — Retrieval Evaluation")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Top-K: {TOP_K}")
    print("=" * 70)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception as e:
        print(f"\n  ✗ ChromaDB collection not found. Run rag_ingest.py first.")
        print(f"    Error: {e}")
        return
    
    total = len(TEST_QUESTIONS)
    hits = 0
    results = []
    
    print(f"\n  Running {total} test queries...\n")
    print(f"  {'#':>3}  {'Topic':<45}  {'Result':<8}  {'Retrieved Classes'}")
    print(f"  {'─'*3}  {'─'*45}  {'─'*8}  {'─'*30}")
    
    for i, tq in enumerate(TEST_QUESTIONS, 1):
        query = tq["question"]
        expected = tq["expected_class"]
        topic = tq["topic"]
        
        # Query ChromaDB
        query_results = collection.query(
            query_texts=[query],
            n_results=TOP_K,
        )
        
        # Extract class indices from retrieved metadata
        retrieved_classes = []
        retrieved_sources = []
        if query_results["metadatas"] and query_results["metadatas"][0]:
            for meta in query_results["metadatas"][0]:
                cls = meta.get("class_index", -1)
                retrieved_classes.append(cls)
                retrieved_sources.append(meta.get("source", "Unknown"))
        
        # Check if expected class is in retrieved results
        is_hit = expected in retrieved_classes
        if is_hit:
            hits += 1
        
        status = "✓ HIT" if is_hit else "✗ MISS"
        classes_str = str(retrieved_classes)
        
        print(f"  {i:>3}  {topic:<45}  {status:<8}  {classes_str}")
        
        results.append({
            "question_num": i,
            "topic": topic,
            "question": query,
            "expected_class": expected,
            "retrieved_classes": retrieved_classes,
            "retrieved_sources": retrieved_sources[:3],  # Top 3 sources
            "is_hit": is_hit,
        })
    
    # Calculate and display hit rate
    hit_rate = (hits / total) * 100
    
    print(f"\n  {'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'='*70}")
    print(f"  Total Questions:   {total}")
    print(f"  Hits:              {hits}")
    print(f"  Misses:            {total - hits}")
    print(f"  Hit Rate:          {hit_rate:.1f}%")
    print(f"  {'='*70}")
    
    # Classification of hit rate
    if hit_rate >= 90:
        grade = "Excellent"
    elif hit_rate >= 75:
        grade = "Good"
    elif hit_rate >= 60:
        grade = "Acceptable"
    else:
        grade = "Needs Improvement"
    
    print(f"  Grade:             {grade}")
    print(f"  {'='*70}")
    
    # Save results to JSON
    evaluation_output = {
        "evaluation_date": datetime.now().isoformat(),
        "configuration": {
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "collection_name": COLLECTION_NAME,
        },
        "summary": {
            "total_questions": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate_percent": round(hit_rate, 1),
            "grade": grade,
        },
        "detailed_results": results,
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to: {RESULTS_FILE}")
    
    # Print any misses for analysis
    misses = [r for r in results if not r["is_hit"]]
    if misses:
        print(f"\n  ── Missed Questions Analysis ──")
        for miss in misses:
            print(f"  Q{miss['question_num']}: {miss['topic']}")
            print(f"    Expected class: {miss['expected_class']}")
            print(f"    Retrieved: {miss['retrieved_classes']}")
            print()
    
    return evaluation_output


if __name__ == "__main__":
    evaluate_retrieval()
