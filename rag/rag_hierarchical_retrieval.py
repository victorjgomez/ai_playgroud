from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Sample topic-segmented chunks
topic_chunks = [
    {"topic": "Artificial Intelligence", "text": "Artificial Intelligence (AI) is transforming industries."},
    {"topic": "Finance", "text": "In finance, AI is used to predict market trends and automate trading."},
    {"topic": "Healthcare", "text": "AI helps diagnose diseases and recommend treatments in healthcare."},
    {"topic": "Machine Learning", "text": "Machine learning allows systems to learn from data without programming."},
    {"topic": "Natural Language Processing", "text": "NLP is used in chatbots and virtual assistants."}
]

# Step 2: Prepare model and embed topics and chunks
model = SentenceTransformer("all-MiniLM-L6-v2")

# Unique topics
unique_topics = list(set(chunk["topic"] for chunk in topic_chunks))
topic_embeddings = model.encode(unique_topics)

# Embed all chunks
chunk_embeddings = model.encode([chunk["text"] for chunk in topic_chunks])

# Step 3: Hierarchical retrieval function
def hierarchical_rag(query, top_k=1):
    query_embedding = model.encode([query])[0]

    # Stage 1: Match query to topic
    topic_sim = cosine_similarity([query_embedding], topic_embeddings)[0]
    best_topic_idx = np.argmax(topic_sim)
    selected_topic = unique_topics[best_topic_idx]

    # Stage 2: Search within chunks of that topic
    matching_chunks = [
        (i, chunk) for i, chunk in enumerate(topic_chunks) if chunk["topic"] == selected_topic
    ]
    if not matching_chunks:
        return None, []

    matching_indices = [i for i, _ in matching_chunks]
    matching_texts = [chunk["text"] for _, chunk in matching_chunks]
    matching_embeddings = [chunk_embeddings[i] for i in matching_indices]

    # Compute similarity and rank
    sim_scores = cosine_similarity([query_embedding], matching_embeddings)[0]
    ranked = sorted(zip(sim_scores, matching_texts), key=lambda x: -x[0])

    return selected_topic, ranked[:top_k]

# Step 4: Test
query = "How do computers learn from experience?"
topic, results = hierarchical_rag(query, top_k=2)

print(f"\n🔍 Query: {query}")
print(f"🎯 Matched Topic: {topic}")
print("📄 Top Relevant Chunk(s):")
for i, (score, text) in enumerate(results):
    print(f"{i+1}. Score: {score:.4f} | Text: {text}")
