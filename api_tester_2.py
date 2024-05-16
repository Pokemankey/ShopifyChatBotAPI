import requests


url = 'http://localhost:8000/llm-stream/'
headers = {'Content-Type': 'application/json'}
data = '{"query": "green visa"}'

response = requests.post(url, headers=headers, data=data, stream=True)
if response.ok:
    for chunk in response.iter_content(chunk_size=1024):
        print(chunk.decode())
