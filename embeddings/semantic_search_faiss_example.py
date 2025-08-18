# pip install sentence-transformers faiss-cpu numpy
# (use 'faiss-gpu' if you have CUDA)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

docs = [
    "Order status can be checked on your account page.",
    "Our refund policy covers 30 days after purchase.",
    "Shipping is free for orders over $50 in the US.",
    "You can reset your password from the login screen.",
    "International shipping options are available at checkout.",
    "Contact support via chat or email for billing issues."
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

# Build index
doc_vecs = embed(docs)                      # (N, D)
d = doc_vecs.shape[1]
index = faiss.IndexFlatIP(d)                # inner product == cosine since vectors are normalized
index.add(doc_vecs)                         # add all vectors

# Optional: persist index + raw docs
faiss.write_index(index, "docs.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)

# Query
def search(query: str, k: int = 3):
    q = embed([query])                       # (1, D)
    sims, idx = index.search(q, k)           # sims: (1, k), idx: (1, k)
    return [(int(i), float(s), docs[i]) for i, s in zip(idx[0], sims[0])]

# Try it
for i, score, text in search("refund for my purchase", k=3):
    print(f"{score:.3f} | {text}")
