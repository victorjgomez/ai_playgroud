import openai

openai.api_key = "your-api-key"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="AI: Hello, how can I help you today?\nHuman:",
    max_tokens=100,
    stop=["\n", "Human:"]
)

print(response.choices[0].text.strip())
