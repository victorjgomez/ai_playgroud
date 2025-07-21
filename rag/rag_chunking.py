import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import textwrap

# Step 1: Long document (simulate a big file)
long_document = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. 
AI has applications in healthcare, finance, robotics, and more. In healthcare, AI can analyze medical data to assist in diagnostics and personalized medicine. 
In finance, it can detect fraud and automate trading. Robotics combines AI with mechanical components to build smart systems. 
Machine Learning (ML), a subset of AI, uses statistical techniques to give machines the ability to learn from data without being explicitly programmed. 
Natural Language Processing (NLP) is another subfield that focuses on interaction between computers and humans using natural language.
"""

# Step 2: Chunk the document into small paragraphs (e.g., 30 words each)
def chunk_text(text: str, max_words: int = 30) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = chunk_text(long_document)
print("Chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")

# Step 3: Embed the chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Step 4: Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 5: Simulate a user query
query = "What is machine learning?"
query_embedding = model.encode([query])

# Step 6: Retrieve top-k similar chunks
k = 2
distances, indices = index.search(np.array(query_embedding), k)

# Step 7: Display the relevant context
retrieved_context = "\n".join([chunks[i] for i in indices[0]])
print("\n📄 Retrieved Context:")
print(retrieved_context)

# You could now pass `retrieved_context` + `query` to a language model
