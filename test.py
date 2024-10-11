import requests

url = "http://localhost:7777/predict"

data = {"beatmap_ids": [4547644, 4547645, 4547646]}

response = requests.post(url, json=data)

print(response.json()["4547644"])