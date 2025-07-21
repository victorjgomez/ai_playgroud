from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import torch

# Step 1: Knowledge base
documents = [
    "The capital of France is Paris.",
    "Python is a programming language that emphasizes readability.",
    "The Eiffel Tower is located in Paris.",
    "Machine learning enables computers to learn from data.",
]

# Step 2: Create embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# Step 3: Store in FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Step 4: Load a local text generation model (small to fit in memory)
generator_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
generator_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")


# Step 5: RAG function (Retrieval + Generation)
def rag_local(query, top_k=2):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve top-k relevant context
    context = "\n".join([documents[i] for i in indices[0]])

    # Build prompt for local model
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = generator_tokenizer.encode(prompt, return_tensors="pt")

    outputs = generator_model.generate(
        inputs,
        max_new_tokens=50,
        pad_token_id=generator_tokenizer.eos_token_id
    )

    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()


# Step 6: Test it
query = "Where is the Eiffel Tower?"
answer = rag_local(query)
print("Answer:", answer)
