import requests

url = "http://localhost:7777/predict"

data = {"beatmap_id": 4547644}

response = requests.post(url, json=data)
                        
print(response.json())