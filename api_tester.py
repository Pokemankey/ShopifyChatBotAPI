import requests


url = 'http://localhost:8000/llm-query/'
headers = {'Content-Type': 'application/json'}
data = '{"query": "Tell me about your team"}'

response = requests.post(url, headers=headers, data=data)

print(response.json())
