import requests
import json

def chat_with_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()['response']

# 测试
print(chat_with_ollama("你好，介绍一下自己"))