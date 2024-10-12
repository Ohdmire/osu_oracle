import requests

url = "http://localhost:7777/predict"

data = {"beatmap_ids": [4547644, 4583961]}

response = requests.post(url, json=data)

for beatmap_id, predictions in response.json().items():
    print(beatmap_id)
    print(predictions)