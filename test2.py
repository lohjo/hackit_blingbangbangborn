import os, requests

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Hello"}]
}

r = requests.post(url, headers=headers, json=data)
print(r.status_code, r.text)
