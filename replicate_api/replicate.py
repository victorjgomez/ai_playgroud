"""
Replicate AI API - Python Examples
Complete guide with various model types and use cases
"""

import replicate
import os
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# ========================================
# 1. SETUP AND AUTHENTICATION
# ========================================

# Set your API token (get it from: https://replicate.com/account/api-tokens)
# Method 1: Environment variable (recommended)
# os.environ["REPLICATE_API_TOKEN"] = "r8_your_token_here"


# Method 2: Direct client initialization
# client = replicate.Client(api_token="r8_your_token_here")

# ========================================
# 2. IMAGE GENERATION EXAMPLES
# ========================================

def generate_image_flux():
    """Generate an image using FLUX model"""
    try:
        print("Generating image with FLUX...")
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": "a futuristic city at sunset, cyberpunk style, neon lights",
                "aspect_ratio": "16:9",
                "output_format": "png"
            }
        )

        # Save the generated image
        with open('flux_output.png', 'wb') as f:
            f.write(output[0].read())
        print("Image saved as flux_output.png")

        return output
    except Exception as e:
        print(f"Error generating image: {e}")


def generate_image_sdxl():
    """Generate an image using Stable Diffusion XL"""
    try:
        print("Generating image with SDXL...")
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": "a majestic dragon flying over mountains, fantasy art",
                "negative_prompt": "blurry, low quality, distorted",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        )

        # Handle multiple outputs if generated
        for idx, file_output in enumerate(output):
            with open(f'sdxl_output_{idx}.png', 'wb') as f:
                f.write(file_output.read())
        print("SDXL images saved")

        return output
    except Exception as e:
        print(f"Error generating SDXL image: {e}")


# ========================================
# 3. TEXT GENERATION EXAMPLES
# ========================================

def generate_text_llama():
    """Generate text using Llama model"""
    try:
        print("Generating text with Llama...")
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e6",
            input={
                "prompt": "Explain the concept of machine learning in simple terms:",
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        )

        result = "".join(output)
        print("Generated text:", result)
        return result
    except Exception as e:
        print(f"Error generating text: {e}")


def stream_text_generation():
    """Stream text generation in real-time"""
    try:
        print("Streaming text generation...")
        iterator = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e6",
            input={
                "prompt": "Write a short story about a robot discovering emotions:",
                "max_new_tokens": 300,
                "temperature": 0.8
            }
        )

        print("Story output:")
        full_text = ""
        for text in iterator:
            print(text, end="", flush=True)
            full_text += text

        print("\n\nGeneration complete!")
        return full_text
    except Exception as e:
        print(f"Error streaming text: {e}")


# ========================================
# 4. VISION MODEL EXAMPLES
# ========================================

def analyze_image_with_llava():
    """Use LLaVA model to analyze an image"""
    try:
        print("Analyzing image with LLaVA...")

        # Using a URL (more efficient for large files)
        image_url = "https://example.com/sample_image.jpg"

        # Or using a local file
        # with open("sample_image.jpg", "rb") as image_file:
        #     image = image_file

        output = replicate.run(
            "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
            input={
                "image": image_url,
                "prompt": "Describe this image in detail. What do you see?"
            }
        )

        print("Image analysis:", output)
        return output
    except Exception as e:
        print(f"Error analyzing image: {e}")


def image_to_text_blip():
    """Generate captions for images using BLIP"""
    try:
        print("Generating image caption...")
        output = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={
                "image": "https://example.com/your_image.jpg",
                "task": "image_captioning"
            }
        )

        print("Generated caption:", output)
        return output
    except Exception as e:
        print(f"Error generating caption: {e}")


# ========================================
# 5. AUDIO GENERATION EXAMPLES
# ========================================

def generate_music():
    """Generate music using MusicGen"""
    try:
        print("Generating music...")
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": "upbeat electronic dance music with synthesizer melodies",
                "duration": 10,
                "temperature": 0.8,
                "top_k": 250,
                "top_p": 0.0
            }
        )

        # Save the generated audio
        with open('generated_music.wav', 'wb') as f:
            f.write(output.read())
        print("Music saved as generated_music.wav")

        return output
    except Exception as e:
        print(f"Error generating music: {e}")


# ========================================
# 6. ERROR HANDLING AND BEST PRACTICES
# ========================================

def robust_api_call(model_name, input_params, max_retries=3):
    """Make a robust API call with error handling and retries"""
    import time

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")
            output = replicate.run(model_name, input=input_params)
            print("API call successful!")
            return output

        except replicate.exceptions.ReplicateError as e:
            print(f"Replicate API error: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to complete request.")
                raise

        except Exception as e:
            print(f"Unexpected error: {e}")
            break


# ========================================
# 7. BATCH PROCESSING EXAMPLE
# ========================================

def batch_image_generation(prompts):
    """Generate multiple images from a list of prompts"""
    results = []

    for i, prompt in enumerate(prompts):
        try:
            print(f"Generating image {i + 1}/{len(prompts)}: {prompt[:50]}...")

            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={"prompt": prompt}
            )

            filename = f'batch_output_{i + 1}.png'
            with open(filename, 'wb') as f:
                f.write(output[0].read())

            results.append({
                "prompt": prompt,
                "filename": filename,
                "success": True
            })

        except Exception as e:
            print(f"Failed to generate image for prompt {i + 1}: {e}")
            results.append({
                "prompt": prompt,
                "filename": None,
                "success": False,
                "error": str(e)
            })

    return results


# ========================================
# 8. MAIN EXECUTION EXAMPLE
# ========================================

if __name__ == "__main__":
    # Make sure to set your API token before running
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("Please set your REPLICATE_API_TOKEN environment variable")
        exit(1)

    print("=== Replicate AI API Examples ===\n")

    # Example 1: Generate a single image
    print("1. Generating image...")
    generate_image_flux()

    # Example 2: Stream text generation
    print("\n2. Streaming text generation...")
    stream_text_generation()

    # Example 3: Batch processing
    print("\n3. Batch image generation...")
    prompts = [
        "a serene lake at dawn",
        "a bustling marketplace in Morocco",
        "a space station orbiting Earth"
    ]
    batch_results = batch_image_generation(prompts)
    print(f"Batch completed: {sum(1 for r in batch_results if r['success'])}/{len(batch_results)} successful")

    print("\nAll examples completed!")


# ========================================
# 9. UTILITY FUNCTIONS
# ========================================

def list_available_models():
    """List some popular models available on Replicate"""
    popular_models = {
        "Image Generation": [
            "black-forest-labs/flux-schnell",
            "stability-ai/sdxl",
            "stability-ai/stable-diffusion"
        ],
        "Text Generation": [
            "meta/llama-2-70b-chat",
            "anthropic/claude-3-sonnet",
            "mistralai/mixtral-8x7b-instruct-v0.1"
        ],
        "Vision": [
            "yorickvp/llava-13b",
            "salesforce/blip"
        ],
        "Audio": [
            "meta/musicgen",
            "suno-ai/bark"
        ]
    }

    for category, models in popular_models.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")


def get_prediction_status(prediction_id):
    """Check the status of a prediction"""
    try:
        prediction = replicate.predictions.get(prediction_id)
        print(f"Status: {prediction.status}")
        if prediction.status == "succeeded":
            return prediction.output
        elif prediction.status == "failed":
            print(f"Error: {prediction.error}")
        return None
    except Exception as e:
        print(f"Error checking prediction: {e}")
        return None