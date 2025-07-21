from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "I love cats.",
    "I adore kittens.",
    "I enjoy pizza.",
    "The sky is blue."
]

# Generate embeddings (vectors)
embeddings = model.encode(sentences)

# Compare each sentence to the first one using cosine similarity
base_vector = embeddings[0]  # Vector of "I love cats."

print("Base sentence:", sentences[0])
print("\nSimilarity with other sentences:")

for i in range(1, len(sentences)):
    similarity = cosine_similarity([base_vector], [embeddings[i]])[0][0]
    print(f"- {sentences[i]} → Similarity: {similarity:.4f}")
