from ollama.ollama_request_class import OllamaRequest

ollama_request = OllamaRequest()

response_not_stop = ollama_request.request(query="What do you know about Dominican Republic?")

print("------- Without Stop ----------")
print(response_not_stop)

print("------- With Stop ----------")
stop=["\n", "Tourism"]
response = ollama_request.request(query="What do you know about Dominican Republic?", stop=stop)

print(response)
