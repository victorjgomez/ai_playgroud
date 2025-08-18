# pip install sentence-transformers numpy
from sentence_transformers import SentenceTransformer
import numpy as np

# 1) Categories and example texts
categories = {
    "Sports": [
        "The team won the championship after a tough match",
        "Olympic athletes prepare for the games",
        "The football player scored a hat-trick in the match"
    ],
    "Politics": [
        "The government passed a new healthcare reform bill",
        "The president met with foreign leaders to discuss trade",
        "Senators debated the proposed tax legislation"
    ],
    "Technology": [
        "A new AI model has been released by the research team",
        "Tech companies are investing in quantum computing",
        "The smartphone update improves battery life"
    ]
}

# 2) Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

# 3) Create category centroid embeddings
category_embeddings = {}
for label, examples in categories.items():
    emb = embed(examples)
    category_embeddings[label] = np.mean(emb, axis=0)  # centroid of examples

# 4) Classification function using cosine similarity
def classify(text: str):
    query_vec = embed([text])[0]
    scores = {label: float(np.dot(query_vec, centroid))
              for label, centroid in category_embeddings.items()}
    best_label = max(scores, key=scores.get)
    return best_label, scores

# 5) Test it
test_texts = [
    "The basketball tournament starts next week",
    "Parliament is voting on the climate change bill",
    "Google announced a breakthrough in AI research"
]

for txt in test_texts:
    label, score_dict = classify(txt)
    print(f"Text: {txt}")
    print(f"Predicted: {label}")
    print(f"Scores: {score_dict}\n")
