import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Sample knowledge base
documents = [
    {
        "title": "AI Overview",
        "topic": "Artificial Intelligence",
        "content": """Artificial Intelligence (AI) simulates human intelligence in machines. It has applications in many fields such as robotics, healthcare, and finance."""
    },
    {
        "title": "Machine Learning Basics",
        "topic": "Machine Learning",
        "content": """Machine Learning (ML) is a subset of AI. It allows computers to learn from data using algorithms without being explicitly programmed."""
    },
    {
        "title": "Natural Language Processing",
        "topic": "NLP",
        "content": """Natural Language Processing (NLP) is a branch of AI that deals with the interaction between computers and human language."""
    }
]

# Step 1: Chunk each document by topic (in this case, already well separated)
def prepare_chunks_with_metadata(docs: List[Dict]):
    chunks = []
    metadatas = []
    chunk_id = 0
    for doc in docs:
        # You could split by paragraphs here if `doc["content"]` is large
        chunk_text = doc["content"].strip()
        chunks.append(chunk_text)
        metadatas.append({
            "chunk_id": chunk_id,
            "title": doc["title"],
            "topic": doc["topic"],
            "text": chunk_text
        })
        chunk_id += 1
    return chunks, metadatas

chunks, metadata = prepare_chunks_with_metadata(documents)

# Step 2: Embed the chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Step 3: Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 4: Search function with metadata
def rag_query_with_metadata(query: str, top_k: int = 2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results

# Step 5: Test
query = "How do machines learn from data?"
results = rag_query_with_metadata(query)

print(f"\n🔎 Query: {query}")
print("📄 Top Retrieved Chunks with Metadata:\n")
for result in results:
    print(f"🔹 Title: {result['title']}")
    print(f"🔸 Topic: {result['topic']}")
    print(f"📄 Text: {result['text']}\n")
