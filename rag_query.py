"""
rag_query.py — RAG advisory query module for Orbit-IQ.

After the CNN predicts a disease class, this module:
  1. Retrieves the top-k most relevant passages from ChromaDB
  2. Uses a LangChain RetrievalQA chain to generate grounded advice
  3. Calls the Gemini API to produce advisory text from retrieved passages only
  4. Returns the advice along with cited sources
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# LangChain imports — used for the retrieval chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "plant_disease_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of passages to retrieve
GEMINI_MODEL = "gemini-2.0-flash"


# ── Disease class names (matches CNN.py idx_to_classes) ──────────────
IDX_TO_DISEASE = {
    0: "Apple Scab", 1: "Apple Black Rot", 2: "Apple Cedar Apple Rust",
    3: "Apple (Healthy)", 4: "Background Without Leaves",
    5: "Blueberry (Healthy)", 6: "Cherry Powdery Mildew",
    7: "Cherry (Healthy)", 8: "Corn Cercospora Leaf Spot / Gray Leaf Spot",
    9: "Corn Common Rust", 10: "Corn Northern Leaf Blight",
    11: "Corn (Healthy)", 12: "Grape Black Rot",
    13: "Grape Esca / Black Measles",
    14: "Grape Leaf Blight / Isariopsis Leaf Spot",
    15: "Grape (Healthy)",
    16: "Orange Huanglongbing / Citrus Greening",
    17: "Peach Bacterial Spot", 18: "Peach (Healthy)",
    19: "Pepper Bell Bacterial Spot", 20: "Pepper Bell (Healthy)",
    21: "Potato Early Blight", 22: "Potato Late Blight",
    23: "Potato (Healthy)", 24: "Raspberry (Healthy)",
    25: "Soybean (Healthy)", 26: "Squash Powdery Mildew",
    27: "Strawberry Leaf Scorch", 28: "Strawberry (Healthy)",
    29: "Tomato Bacterial Spot", 30: "Tomato Early Blight",
    31: "Tomato Late Blight", 32: "Tomato Leaf Mold",
    33: "Tomato Septoria Leaf Spot",
    34: "Tomato Spider Mites / Two-Spotted Spider Mite",
    35: "Tomato Target Spot",
    36: "Tomato Yellow Leaf Curl Virus",
    37: "Tomato Mosaic Virus", 38: "Tomato (Healthy)",
}

# Healthy class indices
HEALTHY_CLASSES = {3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38}


# ── Prompt Template ───────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an expert plant pathologist providing advice to farmers and gardeners.

Based ONLY on the following retrieved passages from agricultural extension sources, provide a comprehensive advisory for the detected plant condition: **{disease_name}**.

Retrieved Passages:
{context}

Instructions:
- Base your advice STRICTLY on the information in the retrieved passages above.
- Do NOT add information from outside these passages.
- Structure your response with clear sections:
  1. **Overview**: Brief summary of the condition
  2. **Key Symptoms**: What to look for
  3. **Recommended Actions**: Specific steps the farmer/gardener should take
  4. **Prevention Tips**: How to prevent this in the future
- If the plant is healthy, provide care and maintenance tips instead.
- Use clear, practical language that a farmer or home gardener can follow.
- Keep the response concise but thorough (200-350 words).

Advisory:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["disease_name", "context"],
    template=RAG_PROMPT_TEMPLATE,
)


def _get_langchain_retriever():
    """
    Build a LangChain retriever backed by the existing ChromaDB collection.
    This is the LangChain integration component.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever, vectorstore


def _get_gemini_llm():
    """Initialize the Gemini LLM via LangChain."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it in .env or as an environment variable."
        )
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1024,
    )
    return llm


def query_rag(class_index: int) -> dict:
    """
    Main RAG query function. Given a CNN prediction class index:
    1. Retrieve relevant passages from ChromaDB
    2. Use LangChain RetrievalQA chain with Gemini to generate advice
    3. Return advice + sources
    
    Args:
        class_index: The predicted class index from the CNN model (0-38)
    
    Returns:
        dict with keys:
            - "advice": str — The generated advisory text
            - "sources": list[dict] — List of source documents with metadata
            - "disease_name": str — The disease name
            - "is_healthy": bool — Whether the plant is healthy
            - "error": str | None — Error message if something went wrong
    """
    disease_name = IDX_TO_DISEASE.get(class_index, f"Unknown Class {class_index}")
    is_healthy = class_index in HEALTHY_CLASSES
    
    result = {
        "advice": "",
        "sources": [],
        "disease_name": disease_name,
        "is_healthy": is_healthy,
        "error": None,
    }
    
    try:
        # Build the query based on whether the plant is healthy or diseased
        if is_healthy:
            query = (
                f"Best practices for maintaining healthy {disease_name.split('(')[0].strip()} plants. "
                f"Care tips, watering, fertilization, and disease prevention."
            )
        elif class_index == 4:  # Background without leaves
            result["advice"] = (
                "The uploaded image appears to be a background without any plant leaves. "
                "Please upload a clear image of an individual plant leaf for accurate disease detection. "
                "For best results, photograph the leaf against a plain background with good lighting."
            )
            return result
        else:
            query = (
                f"What is {disease_name}? Symptoms, causes, treatment, management, "
                f"and prevention of {disease_name} in plants."
            )
        
        # ── LangChain Retrieval Chain ─────────────────────────────────
        retriever, vectorstore = _get_langchain_retriever()
        llm = _get_gemini_llm()
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            result["advice"] = f"No detailed information found for {disease_name}. Please consult a local agricultural extension office."
            return result
        
        # Build context from retrieved passages
        context_parts = []
        seen_sources = set()
        for doc in retrieved_docs:
            context_parts.append(doc.page_content)
            source_key = (
                doc.metadata.get("source", "Unknown"),
                doc.metadata.get("url", ""),
            )
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                result["sources"].append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "filename": doc.metadata.get("filename", ""),
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build the LangChain RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": RAG_PROMPT,
            },
        )
        
        # Run the chain
        chain_result = qa_chain.invoke({"query": query, "disease_name": disease_name})
        
        result["advice"] = chain_result.get("result", "Unable to generate advice.")
        
        # Collect sources from chain result as well
        if "source_documents" in chain_result:
            for doc in chain_result["source_documents"]:
                source_key = (
                    doc.metadata.get("source", "Unknown"),
                    doc.metadata.get("url", ""),
                )
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    result["sources"].append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "url": doc.metadata.get("url", ""),
                        "filename": doc.metadata.get("filename", ""),
                    })
        
    except ValueError as e:
        result["error"] = str(e)
        result["advice"] = (
            f"⚠️ RAG Advisory Unavailable: {str(e)}. "
            f"The basic diagnosis is still shown above."
        )
    except Exception as e:
        result["error"] = str(e)
        result["advice"] = (
            f"⚠️ Could not generate AI advisory at this time. "
            f"Error: {str(e)[:200]}. The basic diagnosis is still shown above."
        )
    
    return result


def query_rag_simple(class_index: int) -> dict:
    """
    Simplified RAG query that retrieves passages from ChromaDB directly
    (without LangChain or Gemini). Useful as a fallback when the API key
    is not available.
    
    Returns:
        dict with "passages" and "sources"
    """
    disease_name = IDX_TO_DISEASE.get(class_index, f"Unknown Class {class_index}")
    
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    if class_index in HEALTHY_CLASSES:
        query = f"Best practices for maintaining healthy {disease_name} plants care tips"
    else:
        query = f"{disease_name} symptoms treatment management prevention"
    
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        where={"class_index": class_index},
    )
    
    passages = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    sources = []
    seen = set()
    for meta in metadatas:
        key = (meta.get("source", ""), meta.get("url", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": meta.get("source", "Unknown"),
                "url": meta.get("url", ""),
            })
    
    return {
        "disease_name": disease_name,
        "passages": passages,
        "sources": sources,
    }


# ── CLI Testing ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    test_class = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    print(f"\nQuerying RAG for class {test_class}...")
    print("=" * 60)
    
    # Try full RAG first, fall back to simple
    try:
        result = query_rag(test_class)
        print(f"Disease: {result['disease_name']}")
        print(f"Healthy: {result['is_healthy']}")
        print(f"\nAdvisory:\n{result['advice']}")
        print(f"\nSources:")
        for src in result["sources"]:
            print(f"  - {src['source']}")
            if src.get("url"):
                print(f"    {src['url']}")
        if result["error"]:
            print(f"\nError: {result['error']}")
    except Exception as e:
        print(f"Full RAG failed ({e}), using simple retrieval...")
        result = query_rag_simple(test_class)
        print(f"Disease: {result['disease_name']}")
        print(f"\nRetrieved Passages:")
        for i, passage in enumerate(result["passages"], 1):
            print(f"\n--- Passage {i} ---")
            print(passage[:300] + "...")
        print(f"\nSources:")
        for src in result["sources"]:
            print(f"  - {src['source']}")
