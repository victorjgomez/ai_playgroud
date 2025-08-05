from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

load_dotenv()

# Initialize the client with your API token
api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")
client = InferenceClient(api_key=api_key)


# Alternative: Set token as environment variable
# export HF_API_TOKEN="your_token"
# client = InferenceClient()

# Example 1: Text Generation
def text_generation_example():
    response = client.text_generation(
        prompt="The future of AI is",
        #model="microsoft/DialoGPT-medium",
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9
    )
    print("Generated text:", response)


# Example 2: Text Classification
def text_classification_example():
    response = client.text_classification(
        text="I love this product! It's amazing!",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    print("Classification result:", response)


# Example 3: Question Answering
def question_answering_example():
    response = client.question_answering(
        question="What is machine learning?",
        context="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        model="distilbert-base-cased-distilled-squad"
    )
    print("Answer:", response)


# Example 4: Summarization
def summarization_example():
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """

    response = client.summarization(
        text=text,
        model="facebook/bart-large-cnn",
        max_length=50,
        min_length=10
    )
    print("Summary:", response)


# Example 5: Translation
def translation_example():
    response = client.translation(
        text="Hello, how are you?",
        model="Helsinki-NLP/opus-mt-en-fr"
    )
    print("Translation:", response)


# Example 6: Image Classification
def image_classification_example():
    # Using a local image file
    with open("path/to/your/image.jpg", "rb") as f:
        response = client.image_classification(
            image=f,
            model="google/vit-base-patch16-224"
        )
    print("Image classification:", response)


# Example 7: Text-to-Image Generation
def text_to_image_example():
    response = client.text_to_image(
        prompt="A beautiful sunset over mountains",
        model="stabilityai/stable-diffusion-2-1"
    )
    # Save the generated image
    with open("generated_image.png", "wb") as f:
        f.write(response)
    print("Image generated and saved as 'generated_image.png'")


# Example 8: Conversational AI
def conversational_example():
    response = client.conversational(
        text="What's the weather like today?",
        #model="microsoft/DialoGPT-medium",
        past_user_inputs=["Hello"],
        generated_responses=["Hi there! How can I help you?"]
    )
    print("Conversation response:", response)


# Example 9: Feature Extraction
def feature_extraction_example():
    response = client.feature_extraction(
        text="This is a sample sentence for feature extraction.",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Feature vector shape:", len(response[0]) if response else "No features")


# Example 10: Using Custom Endpoints
def custom_endpoint_example():
    # For deployed models on Hugging Face Inference Endpoints
    endpoint_client = InferenceClient(
        model="https://your-endpoint-url.aws.endpoints.huggingface.cloud"
    )

    response = endpoint_client.post(
        json={"inputs": "Your input text here"}
    )
    print("Custom endpoint response:", response)


# Example 11: Async Usage
import asyncio


async def async_example():
    async_client = InferenceClient(api_key=api_key)

    response = await async_client.text_generation(
        prompt="The future of technology is",
        model="gpt2",
        max_new_tokens=30
    )
    print("Async response:", response)


# Example 12: Error Handling
def error_handling_example():
    try:
        response = client.text_generation(
            prompt="Test prompt",
            model="invalid-model-name"
        )
    except Exception as e:
        print(f"Error occurred: {e}")


# Run examples
if __name__ == "__main__":
    # Make sure to set your API token first
    # You can get it from https://huggingface.co/settings/tokens

    print("Running Hugging Face Inference SDK Examples...")

    # Uncomment the examples you want to run:
    # text_generation_example()
    text_classification_example()
    question_answering_example()
    summarization_example()
    translation_example()
    # image_classification_example()
    text_to_image_example()
    conversational_example()
    # feature_extraction_example()
    # error_handling_example()

    # For async example:
    # asyncio.run(async_example())