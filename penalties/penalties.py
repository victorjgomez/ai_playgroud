'''
Penalties in the context of language models (like OpenAI's GPT) are parameters that help control the diversity and repetitiveness of the generated text:
Frequency Penalty (-2.0 to 2.0):
Reduces repetition by penalizing tokens based on how frequently they've appeared
Higher values decrease the likelihood of repeating the same words/phrases
Useful when you want more diverse vocabulary in responses
Presence Penalty (-2.0 to 2.0):
Penalizes tokens that have appeared at all in the text
Higher values encourage the model to talk about new topics
Helps avoid getting stuck on the same subject matter
In your code example, you're testing three scenarios:
No penalties (both set to 0.0)
Frequency penalty of 1.0
Presence penalty of 1.0
These parameters help fine-tune the model's output for different use cases:
Low penalties: Good for factual/technical responses
Higher penalties: Better for creative writing or brainstorming
'''
from ollama.ollama_request_class import OllamaRequest


prompt = "List some animals you might find in a forest:"

ollama_request = OllamaRequest()

# No penalties
response_none = ollama_request.request(prompt=prompt, repeat_penalty=0.0,
                                       presence_penalty=0.0)

# With frequency_penalty
response_freq = ollama_request.request(prompt=prompt, repeat_penalty=1.0,
                                       presence_penalty=0.0)

# With presence_penalty
response_pres = ollama_request.request(prompt=prompt, repeat_penalty=0.0,
                                       presence_penalty=1.0)

print("== No Penalty ==")
print(response_none)

print("\n== Frequency Penalty ==")
print(response_freq)

print("\n== Presence Penalty ==")
print(response_pres)
