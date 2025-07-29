import replicate
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the client
client = replicate.Client()

# Define the model and its input
model_version = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
input_data = {
    "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic lighting",
    "width": 1024,
    "height": 1024,
    "num_outputs": 1,
    "num_inference_steps": 25,
    "guidance_scale": 7.5
}

print(f"Running model: {model_version} with prompt: '{input_data['prompt']}'")

# Run the model using the client
output = client.run(
    model_version,
    input=input_data
)

# Handle the output
if isinstance(output, list) and output:
    print("\nGenerated image URLs:")
    for url in output:
        print(url)
else:
    print(f"Model output: {output}")