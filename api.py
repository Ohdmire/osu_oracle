from fastapi import FastAPI, HTTPException
from test_model import get_json_predictions
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class PredictionRequest(BaseModel):
    beatmap_ids: List[int]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        folders = ["models"]
        beatmap_ids = [str(id) for id in request.beatmap_ids]
        results = get_json_predictions(folders, beatmap_ids)
        
        # 合并每个 beatmap ID 的预测结果
        formatted_results = {}
        for beatmap_id, predictions in results.items():
            merged_prediction = {}
            for prediction in predictions:
                merged_prediction.update(prediction)
            formatted_results[beatmap_id] = merged_prediction
        
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)