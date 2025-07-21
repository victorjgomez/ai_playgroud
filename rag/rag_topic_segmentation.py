from keybert import KeyBERT
from nltk import sent_tokenize
from typing import List, Dict
import numpy as np

# Sample long document
text = """
Artificial Intelligence (AI) is transforming industries. It allows machines to perform tasks like humans.
In finance, AI is used to predict market trends and automate trading.
In healthcare, AI helps diagnose diseases faster and recommend treatments.
Machine Learning (ML), a subfield of AI, allows systems to learn from data.
Deep Learning, a subset of ML, uses neural networks to analyze large datasets.
Natural Language Processing (NLP) is used in virtual assistants and chatbots.
NLP helps computers understand and respond to human language.
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

# if __name__ == '__main__':
#     import nltk
#
#     nltk.download('punkt')
