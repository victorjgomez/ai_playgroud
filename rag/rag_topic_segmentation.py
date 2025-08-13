from keybert import KeyBERT
from nltk import sent_tokenize
from typing import List, Dict
import numpy as np
import nltk

# Get both resources (new NLTK needs punkt_tab too)
nltk.download('punkt', force=True, quiet=True)
nltk.download('punkt_tab', force=True, quiet=True)

# Sample long document
text = """
Artificial Intelligence (AI) is transforming industries across the globe. It enables machines to perform tasks that traditionally required human intelligence, such as problem-solving, decision-making, and understanding natural language. 
AI technologies are becoming more accessible due to advances in computing power, large datasets, and improved algorithms.

In finance, AI is used to predict market trends, detect fraudulent transactions, and automate trading strategies. 
Banks employ AI-powered chatbots to handle customer inquiries, while risk assessment models help institutions make more informed lending decisions. 
Robo-advisors are also leveraging AI to provide personalized investment strategies based on client goals and risk tolerance.

In healthcare, AI assists in diagnosing diseases faster and with higher accuracy. 
Medical imaging systems enhanced with AI can detect anomalies such as tumors in X-rays or MRIs. 
Predictive analytics help hospitals anticipate patient admission rates, while AI-driven drug discovery accelerates the development of new treatments. 
AI is also being used to create virtual health assistants that can provide basic medical advice or remind patients to take medications.

Machine Learning (ML), a subfield of AI, enables systems to improve their performance over time by learning from data. 
It is widely used in recommendation systems, such as those employed by e-commerce platforms and streaming services to suggest products or content tailored to individual preferences. 
ML algorithms are also applied in supply chain optimization, helping companies forecast demand and manage inventory efficiently.

Deep Learning, a specialized branch of ML, uses artificial neural networks inspired by the human brain to analyze vast and complex datasets. 
It has revolutionized fields like computer vision, enabling applications such as facial recognition, autonomous vehicles, and advanced image editing. 
In speech recognition, deep learning models power virtual assistants and real-time translation services, breaking down language barriers worldwide.

Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language. 
It is at the core of chatbots, virtual assistants, and automated transcription tools. 
NLP technologies are used in sentiment analysis to gauge public opinion on social media and in summarization tools that condense large documents into digestible insights. 
Recent advancements in NLP have enabled AI models to engage in more natural and context-aware conversations, making them valuable in education, customer service, and creative writing.

As AI continues to evolve, ethical considerations such as bias, transparency, and data privacy are becoming increasingly important. 
Governments, organizations, and researchers are collaborating to create guidelines and frameworks to ensure AI technologies are used responsibly and for the benefit of society.
"""

# Step 1: Break text into sentences
sentences = sent_tokenize(text)

# Step 2: Use KeyBERT to extract the top topic per sentence
kw_model = KeyBERT()
topics = []
for sentence in sentences:
    keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1)
    topic = keywords[0][0] if keywords else "general"
    topics.append((topic, sentence))

# Step 3: Group sentences by dominant topic (or switch when topic changes)
def segment_by_topic(topic_sentence_pairs: List[tuple]) -> List[Dict]:
    segments = []
    current_topic = topic_sentence_pairs[0][0]
    current_chunk = []

    for topic, sentence in topic_sentence_pairs:
        if topic != current_topic and current_chunk:
            segments.append({
                "topic": current_topic,
                "text": " ".join(current_chunk)
            })
            current_chunk = []
            current_topic = topic
        current_chunk.append(sentence)

    if current_chunk:
        segments.append({
            "topic": current_topic,
            "text": " ".join(current_chunk)
        })

    return segments

# Step 4: Segment and print
topic_chunks = segment_by_topic(topics)

print("\n🔖 Topic Segments:")
for i, chunk in enumerate(topic_chunks):
    print(f"\nChunk {i+1}")
    print(f"Topic: {chunk['topic']}")
    print(f"Text: {chunk['text']}")
