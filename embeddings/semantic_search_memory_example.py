# pip install sentence-transformers numpy
from sentence_transformers import SentenceTransformer
import numpy as np

# 1) Data
docs = [
    "Order status can be checked on your account page.",
    "Our refund policy covers 30 days after purchase.",
    "Shipping is free for orders over $50 in the US.",
    "You can reset your password from the login screen."
]

# 2) Embed with a small, fast model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    # Returns (n, d) float32 matrix
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

doc_vecs = embed(docs)  # (N, D)

# 3) Search: cosine similarity via dot product (already normalized)
def search(query: str, k: int = 3):
    q_vec = embed([query])[0]               # (D,)
    sims = doc_vecs @ q_vec                 # (N,)
    topk_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), docs[i]) for i in topk_idx]

# 4) Try it
for i, score, text in search("How do I get a refund?", k=3):
    print(f"{score:.3f} | {text}")
