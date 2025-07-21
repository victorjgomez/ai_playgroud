from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load model and define documents
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The cat sits on the mat.",
    "Dogs are friendly animals.",
    "I love playing with my kitten.",
    "The sky is bright blue today.",
    "Cats and dogs are common pets."
]

# Step 2: Generate embeddings (vectors)
embeddings = model.encode(documents)

# Step 3: Create FAISS index
dimension = embeddings.shape[1]  # e.g., 384
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(np.array(embeddings))

# Step 4: Perform a query
query = "I adore my pet cat."
query_embedding = model.encode([query])

# Step 5: Search top 3 most similar documents
k = 3
distances, indices = index.search(np.array(query_embedding), k)

# Step 6: Print results
print(f"Query: {query}\n")
print("Top matches:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")
