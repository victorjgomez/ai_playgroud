import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ollama.ollama_request_class import OllamaRequest

# Step 1: Your knowledge base
documents = [
    "The capital of France is Paris.",
    "Python is a programming language that emphasizes readability.",
    "The Eiffel Tower is located in Paris.",
    "Machine learning enables computers to learn from data.",
]

# Step 2: Embed documents
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

# Step 3: Store in FAISS
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

ollama_request = OllamaRequest()

# Step 4: RAG function
def rag_query(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve top-k context
    context = "\n".join([documents[i] for i in indices[0]])

    print(f"context: {context}")

    response = ollama_request.request(query=query, context=context)

    return response

# Step 5: Test the RAG system
query = "Where is the Eiffel Tower?"
answer = rag_query(query)
print("Answer:", answer)
