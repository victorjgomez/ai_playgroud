"""
Hugging Face Inference API Examples using requests library
Direct API calls without the huggingface_hub SDK
"""

# Standard library imports
import os
import json
import time
import base64
from typing import Dict, Any, List, Optional

# Third-party imports
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://api-inference.huggingface.co/models"
TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not TOKEN:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables")

# Common headers for all requests
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


class HuggingFaceAPI:
    """Direct Hugging Face API client using requests"""

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api-inference.huggingface.co/models"

    def make_request(self, model: str, payload: Dict[str, Any],
                     max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
        """Make a request to the HF API with retry logic"""
        url = f"{self.base_url}/{model}"

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading
                    print(f"Model {model} is loading, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                elif response.status_code == 429:
                    # Rate limited
                    print(f"Rate limited, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    return {"error": f"HTTP {response.status_code}: {response.text}"}

            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    def text_generation(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using a language model"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_new_tokens", 50),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
            }
        }
        return self.make_request(model, payload)

    def text_classification(self, model: str, text: str) -> Dict[str, Any]:
        """Classify text sentiment/category"""
        payload = {"inputs": text}
        return self.make_request(model, payload)

    def question_answering(self, model: str, question: str, context: str) -> Dict[str, Any]:
        """Answer questions based on context"""
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        return self.make_request(model, payload)

    def summarization(self, model: str, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text"""
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": kwargs.get("max_length", 130),
                "min_length": kwargs.get("min_length", 30),
                "do_sample": kwargs.get("do_sample", False)
            }
        }
        return self.make_request(model, payload)

    def translation(self, model: str, text: str) -> Dict[str, Any]:
        """Translate text"""
        payload = {"inputs": text}
        return self.make_request(model, payload)

    def feature_extraction(self, model: str, text: str) -> Dict[str, Any]:
        """Extract embeddings/features from text"""
        payload = {"inputs": text}
        return self.make_request(model, payload)

    def image_classification(self, model: str, image_path: str) -> Dict[str, Any]:
        """Classify images"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

            # For image tasks, we need to send binary data
            url = f"{self.base_url}/{model}"
            headers = {"Authorization": f"Bearer {self.token}"}

            with open(image_path, "rb") as image_file:
                response = requests.post(url, headers=headers, data=image_file.read())

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            return {"error": str(e)}

    def text_to_image(self, model: str, prompt: str, **kwargs) -> bytes:
        """Generate images from text"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
                "guidance_scale": kwargs.get("guidance_scale", 7.5)
            }
        }

        url = f"{self.base_url}/{model}"
        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.content  # Returns image bytes
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


# Initialize API client
api = HuggingFaceAPI(TOKEN)


def test_text_generation():
    """Test text generation with different models"""
    print("=== Text Generation ===")

    models = ["gpt2", "microsoft/DialoGPT-medium"]
    prompts = [
        "The future of artificial intelligence is",
        "In a world where robots and humans coexist"
    ]

    for model in models:
        print(f"\nModel: {model}")
        for prompt in prompts[:1]:  # Test first prompt only
            print(f"Prompt: '{prompt}'")
            result = api.text_generation(
                model=model,
                prompt=prompt,
                max_new_tokens=40,
                temperature=0.8
            )

            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    print(f"Generated: {generated_text}")
                else:
                    print(f"Result: {result}")
            print("-" * 50)


def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("=== Sentiment Analysis ===")

    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    texts = [
        "I love this product!",
        "This is terrible, I hate it",
        "It's okay, nothing special",
        "Best purchase ever!",
        "Completely disappointed"
    ]

    for text in texts:
        result = api.text_classification(model, text)
        print(f"Text: '{text}'")

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            if isinstance(result, list):
                for prediction in result:
                    label = prediction.get("label", "Unknown")
                    score = prediction.get("score", 0)
                    print(f"  {label}: {score:.3f}")
            else:
                print(f"Result: {result}")
        print("-" * 40)


def test_question_answering():
    """Test question answering"""
    print("=== Question Answering ===")

    model = "distilbert-base-cased-distilled-squad"
    context = """
    Machine learning is a method of data analysis that automates analytical 
    model building. It is a branch of artificial intelligence (AI) based on 
    the idea that systems can learn from data, identify patterns and make 
    decisions with minimal human intervention.
    """

    questions = [
        "What is machine learning?",
        "What is it based on?",
        "How much human intervention is needed?"
    ]

    for question in questions:
        result = api.question_answering(model, question, context)
        print(f"Question: {question}")

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            answer = result.get("answer", "No answer found")
            score = result.get("score", 0)
            print(f"Answer: {answer} (confidence: {score:.3f})")
        print("-" * 40)


def test_summarization():
    """Test text summarization"""
    print("=== Text Summarization ===")

    model = "facebook/bart-large-cnn"
    text = """
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, 
    and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
    During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
    man-made structure in the world, a title it held for 41 years until the Chrysler Building in 
    New York City was finished in 1930. It was the first structure to reach a height of 300 metres. 
    Due to the addition of a broadcasting antenna at the top of the tower in 1957, it is now taller 
    than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is 
    the second tallest free-standing structure in France after the Millau Bridge.
    """

    result = api.summarization(
        model=model,
        text=text,
        max_length=60,
        min_length=20
    )

    print(f"Original text length: {len(text)} characters")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary_text", "")
            print(f"Summary: {summary}")
            print(f"Summary length: {len(summary)} characters")
        else:
            print(f"Result: {result}")


def test_translation():
    """Test language translation"""
    print("=== Translation ===")

    translations = [
        ("Helsinki-NLP/opus-mt-en-fr", "Hello, how are you today?", "English to French"),
        ("Helsinki-NLP/opus-mt-en-es", "Good morning, have a nice day!", "English to Spanish"),
        ("Helsinki-NLP/opus-mt-en-de", "The weather is beautiful today.", "English to German")
    ]

    for model, text, description in translations:
        print(f"{description}:")
        print(f"Original: {text}")

        result = api.translation(model, text)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            if isinstance(result, list) and len(result) > 0:
                translated = result[0].get("translation_text", "")
                print(f"Translated: {translated}")
            else:
                print(f"Result: {result}")
        print("-" * 40)


def test_feature_extraction():
    """Test feature extraction (embeddings)"""
    print("=== Feature Extraction ===")

    model = "sentence-transformers/all-MiniLM-L6-v2"
    texts = [
        "This is a sample sentence.",
        "Here is another example text.",
        "Machine learning is fascinating."
    ]

    for text in texts:
        result = api.feature_extraction(model, text)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            if isinstance(result, list):
                # Embeddings are usually nested lists
                if len(result) > 0 and isinstance(result[0], list):
                    embedding_dim = len(result[0])
                    print(f"Text: '{text}'")
                    print(f"Embedding dimension: {embedding_dim}")
                    print(f"First 5 values: {result[0][:5]}")
                else:
                    print(f"Unexpected format: {type(result)}")
            else:
                print(f"Result: {result}")
        print("-" * 30)


def test_image_generation():
    """Test text-to-image generation"""
    print("=== Text-to-Image Generation ===")

    model = "runwayml/stable-diffusion-v1-5"
    prompts = [
        "A beautiful sunset over mountains",
        "A cute cat wearing sunglasses"
    ]

    for i, prompt in enumerate(prompts):
        print(f"Generating image for: '{prompt}'")

        try:
            image_bytes = api.text_to_image(
                model=model,
                prompt=prompt,
                num_inference_steps=25
            )

            filename = f"generated_image_{i + 1}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)

            print(f"✓ Image saved as {filename}")

        except Exception as e:
            print(f"✗ Error generating image: {e}")

        print("-" * 40)


def check_api_status():
    """Check if the API is accessible"""
    print("=== API Status Check ===")

    # Simple test with a lightweight model
    test_model = "gpt2"
    test_prompt = "Hello"

    result = api.text_generation(
        model=test_model,
        prompt=test_prompt,
        max_new_tokens=5
    )

    if "error" in result:
        print(f"✗ API Error: {result['error']}")
        return False
    else:
        print("✓ API is accessible")
        print(f"✓ Token is valid: {TOKEN[:10]}...")
        return True


def batch_requests_example():
    """Example of processing multiple requests efficiently"""
    print("=== Batch Processing ===")

    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    texts = [
        "I absolutely love this!",
        "This is the worst thing ever",
        "It's pretty good overall",
        "Not bad, could be better",
        "Amazing quality and service!"
    ]

    results = []
    for i, text in enumerate(texts, 1):
        result = api.text_classification(model, text)
        results.append((text, result))
        print(f"Processed {i}/{len(texts)}: '{text[:30]}...'")

    print("\nBatch Results:")
    for text, result in results:
        print(f"Text: '{text}'")
        if "error" not in result and isinstance(result, list):
            best_prediction = max(result, key=lambda x: x.get("score", 0))
            print(f"Sentiment: {best_prediction['label']} ({best_prediction['score']:.3f})")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("-" * 30)


def main():
    """Main function to run all examples"""
    print("=" * 60)
    print("Hugging Face Direct API Examples using requests")
    print("=" * 60)

    # Check API status first
    if not check_api_status():
        print("Cannot proceed - API is not accessible")
        return

    # Run examples
    print("\n" + "=" * 40)
    test_text_generation()

    print("\n" + "=" * 40)
    test_sentiment_analysis()

    print("\n" + "=" * 40)
    test_question_answering()

    print("\n" + "=" * 40)
    test_summarization()

    print("\n" + "=" * 40)
    test_translation()

    print("\n" + "=" * 40)
    # test_feature_extraction()

    print("\n" + "=" * 40)
    # batch_requests_example()

    # Uncomment to test image generation (requires stable diffusion model)
    # print("\n" + "=" * 40)
    # test_image_generation()

    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()