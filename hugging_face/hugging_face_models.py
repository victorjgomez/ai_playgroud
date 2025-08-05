# Hugging Face Models - Complete Python Guide

# First, install the required packages:
# pip install transformers torch torchvision torchaudio
# pip install datasets accelerate  # Optional but recommended

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    pipeline, Trainer, TrainingArguments
)
from datasets import Dataset
import numpy as np


# =====================================================
# 1. USING PIPELINES (Easiest Way)
# =====================================================

def pipeline_examples():
    """Using pre-built pipelines for common tasks"""

    print("=== Text Classification ===")
    classifier = pipeline("text-classification",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    result = classifier("I love using Hugging Face models!")
    print(result)

    print("\n=== Text Generation ===")
    generator = pipeline("text-generation",
                         model="microsoft/DialoGPT-medium")
    result = generator("Hello, how are you?", max_length=50, num_return_sequences=1)
    print(result[0]['generated_text'])

    print("\n=== Question Answering ===")
    qa_pipeline = pipeline("question-answering",
                           model="distilbert-base-uncased-distilled-squad")
    context = "Hugging Face is a company that develops tools for machine learning."
    question = "What does Hugging Face develop?"
    result = qa_pipeline(question=question, context=context)
    print(f"Answer: {result['answer']}")

    print("\n=== Fill Mask ===")
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    result = fill_mask("The weather today is [MASK].")
    print(result)

    print("\n=== Summarization ===")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """
    result = summarizer(text, max_length=50, min_length=10)
    print(result[0]['summary_text'])


# =====================================================
# 2. USING MODELS AND TOKENIZERS DIRECTLY
# =====================================================

def direct_model_usage():
    """Using models and tokenizers directly for more control"""

    print("=== BERT for Text Classification ===")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    text = "I really enjoy working with machine learning!"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)

    print(f"Text: {text}")
    print(f"Predictions: {predictions}")
    print(f"Predicted class: {predicted_class.item()}")

    print("\n=== GPT-2 Text Generation ===")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The future of artificial intelligence"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


# =====================================================
# 3. WORKING WITH DIFFERENT MODEL TYPES
# =====================================================

def different_model_types():
    """Examples of different model architectures"""

    print("=== BERT (Encoder-only) ===")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    text = "Hello, this is a sample sentence."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    # Get embeddings
    last_hidden_states = outputs.last_hidden_state
    print(f"BERT output shape: {last_hidden_states.shape}")

    print("\n=== DistilBERT (Question Answering) ===")
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    question = "What is machine learning?"
    context = "Machine learning is a subset of artificial intelligence that focuses on algorithms."

    inputs = qa_tokenizer(question, context, return_tensors="pt")
    outputs = qa_model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    print(f"Q: {question}")
    print(f"A: {answer}")


# =====================================================
# 4. FINE-TUNING MODELS
# =====================================================

def fine_tuning_example():
    """Basic fine-tuning example"""

    print("=== Fine-tuning Setup ===")

    # Sample data
    texts = [
        "This movie is great!",
        "I hate this film.",
        "Amazing performance by the actors.",
        "Worst movie ever made.",
        "I love the storyline."
    ]
    labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative

    # Create dataset
    dataset = Dataset.from_dict({"text": texts, "labels": labels})

    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Fine-tuning setup complete. Call trainer.train() to start training.")
    # trainer.train()  # Uncomment to actually train


# =====================================================
# 5. USING CUSTOM MODELS FROM HUG FACE HUB
# =====================================================

def custom_models():
    """Loading and using various models from Hugging Face Hub"""

    print("=== Loading Different Models ===")

    # Sentiment analysis with different model
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    result = sentiment_pipeline("This is an amazing product!")
    print(f"Sentiment: {result}")

    # Translation
    translator = pipeline(
        "translation_en_to_fr",
        model="t5-base"
    )
    result = translator("Hello, how are you today?")
    print(f"Translation: {result}")

    # Text similarity
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        sentences = ["This is a happy day", "Today is a joyful day", "It's raining cats and dogs"]
        embeddings = model.encode(sentences)

        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])
        print(f"Similarities: {similarities}")

    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")


# =====================================================
# 6. OPTIMIZATIONS AND BEST PRACTICES
# =====================================================

def optimizations():
    """Performance optimizations and best practices"""

    print("=== Model Optimizations ===")

    # 1. Using device mapping for large models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Loading model with specific precision
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # 3. Batch processing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = ["Hello world", "How are you?", "Machine learning is fun"]

    # Tokenize in batch
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Move to device if using GPU
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)

    # Generate in batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True)
                       for output in outputs]

    for i, text in enumerate(generated_texts):
        print(f"Input {i + 1}: {texts[i]}")
        print(f"Output {i + 1}: {text}")
        print()


# =====================================================
# 7. UTILITY CLASS FOR EASY MODEL MANAGEMENT
# =====================================================

class HuggingFaceModel:
    """Utility class for managing Hugging Face models"""

    def __init__(self, model_name: str, task: str = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if task:
            self.pipeline = pipeline(task, model=model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)

    def predict(self, text: str, **kwargs):
        """Make predictions using the model"""
        if hasattr(self, 'pipeline'):
            return self.pipeline(text, **kwargs)
        else:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            return outputs

    def batch_predict(self, texts: list, **kwargs):
        """Make batch predictions"""
        if hasattr(self, 'pipeline'):
            return self.pipeline(texts, **kwargs)
        else:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            return outputs


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("🤗 Hugging Face Models Examples")
    print("=" * 50)

    # Run examples (comment out sections you don't want to run)
    pipeline_examples()
    print("\n" + "=" * 50 + "\n")

    direct_model_usage()
    print("\n" + "=" * 50 + "\n")

    different_model_types()
    print("\n" + "=" * 50 + "\n")

    fine_tuning_example()
    print("\n" + "=" * 50 + "\n")

    custom_models()
    print("\n" + "=" * 50 + "\n")

    optimizations()
    print("\n" + "=" * 50 + "\n")

    # Example using the utility class
    print("=== Using Utility Class ===")
    sentiment_model = HuggingFaceModel("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment-analysis")
    result = sentiment_model.predict("I love programming!")
    print(f"Sentiment result: {result}")