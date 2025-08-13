import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:1.5b",
    "prompt": "What's the capital of France?",
    "stream": False,            # 👈 turn off streaming
}

res = requests.post(url, json=payload)
res.raise_for_status()
data = res.json()
print(data["response"])
