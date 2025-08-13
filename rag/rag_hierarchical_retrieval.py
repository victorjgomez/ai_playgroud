from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Sample topic-segmented chunks
topic_chunks = [
    {"topic": "Artificial Intelligence", "text": "Artificial Intelligence (AI) is transforming industries."},
    {"topic": "Artificial Intelligence", "text": "AI enables machines to perform tasks that typically require human intelligence."},
    {"topic": "Artificial Intelligence", "text": "It is widely used in applications such as image recognition, speech processing, and decision-making."},
    {"topic": "Artificial Intelligence", "text": "AI systems can analyze vast amounts of data to uncover patterns and insights."},
    {"topic": "Artificial Intelligence", "text": "The development of AI raises ethical concerns, including bias and job displacement."},
    {"topic": "Finance", "text": "In finance, AI is used to predict market trends and automate trading."},
    {"topic": "Finance", "text": "AI-powered algorithms can analyze historical data to make investment decisions."},
    {"topic": "Finance", "text": "Fraud detection systems leverage AI to identify suspicious transactions."},
    {"topic": "Finance", "text": "AI is also used in credit scoring to assess the creditworthiness of individuals."},
    {"topic": "Finance", "text": "Robo-advisors provide personalized financial advice using AI models."},
    {"topic": "Healthcare", "text": "AI helps diagnose diseases and recommend treatments in healthcare."},
    {"topic": "Healthcare", "text": "Medical imaging systems use AI to detect abnormalities in X-rays and MRIs."},
    {"topic": "Healthcare", "text": "AI-powered chatbots assist patients by answering health-related questions."},
    {"topic": "Healthcare", "text": "Predictive analytics in AI helps identify patients at risk of developing chronic conditions."},
    {"topic": "Healthcare", "text": "AI is also used in drug discovery to accelerate the development of new medications."},
    {"topic": "Machine Learning", "text": "Machine learning allows systems to learn from data without programming."},
    {"topic": "Machine Learning", "text": "It is a subset of AI that focuses on building models that improve over time."},
    {"topic": "Machine Learning", "text": "Supervised learning involves training models on labeled datasets."},
    {"topic": "Machine Learning", "text": "Unsupervised learning is used to find hidden patterns in unlabeled data."},
    {"topic": "Machine Learning", "text": "Reinforcement learning trains models through trial and error to maximize rewards."},
    {"topic": "Natural Language Processing", "text": "NLP is used in chatbots and virtual assistants."},
    {"topic": "Natural Language Processing", "text": "It enables machines to understand and generate human language."},
    {"topic": "Natural Language Processing", "text": "Applications of NLP include sentiment analysis, machine translation, and text summarization."},
    {"topic": "Natural Language Processing", "text": "Speech-to-text systems rely on NLP to convert spoken words into written text."},
    {"topic": "Natural Language Processing", "text": "NLP is also used in search engines to improve the relevance of search results."}
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
query = "How do computers/machines learn from experience?"
topic, results = hierarchical_rag(query, top_k=3)

print(f"\n🔍 Query: {query}")
print(f"🎯 Matched Topic: {topic}")
print("📄 Top Relevant Chunk(s):")
for i, (score, text) in enumerate(results):
    print(f"{i + 1}. Score: {score:.4f} | Text: {text}")
