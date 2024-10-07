from fastapi import FastAPI, HTTPException
from test_model import get_json_predictions, load_model
from pydantic import BaseModel
import json
import uvicorn
import os

app = FastAPI()

class PredictionRequest(BaseModel):
    beatmap_id: int

# 预加载模型
folders = ["models"]
models = []
for folder in folders:
    model_folder = folder + "/"
    model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
    model = [load_model(model_path, compile=False) for model_path in model_paths]
    models.append(model)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        json_predictions = get_json_predictions(folders, str(request.beatmap_id), models)
        
        # 合并所有模型的预测结果
        all_predictions = {}
        for prediction in json_predictions:
            if isinstance(prediction, dict):
                all_predictions.update(prediction)
            else:
                all_predictions.update(json.loads(prediction))
        
        return all_predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)