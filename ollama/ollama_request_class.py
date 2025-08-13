from typing import Optional, List

import requests


class OllamaRequest:

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None):
        self.model = model if model else "deepseek-r1:1.5b"
        self.url = url if url else "http://localhost:11434/api/generate"

    def request(self, query: str, context: Optional[str] = None, temperature: float = 0.7,
                top_p: float = 0.9, repeat_penalty: float = 1.1,
                presence_penalty: float = 0.5, stop: Optional[List[str]] = None) -> str:

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:" if context else query

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # 👈 turn off streaming
            "temperature": temperature,  # creativity control
            "top_p": top_p,  # nucleus sampling
            "repeat_penalty": repeat_penalty,  # discourage repetition
            "presence_penalty": presence_penalty  # encourage introducing new topics
        }

        if stop:
            payload["stop"] = stop

        res = requests.post(self.url, json=payload)
        res.raise_for_status()
        data = res.json()
        #print(data["response"])

        return data["response"]
