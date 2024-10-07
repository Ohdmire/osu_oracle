from fastapi import FastAPI, HTTPException
from test_model import get_json_predictions
from pydantic import BaseModel
import json

app = FastAPI()

class PredictionRequest(BaseModel):
    beatmap_id: int

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        folders = ["models"]
        json_predictions = get_json_predictions(folders, str(request.beatmap_id))
        
        # 合并所有模型的预测结果
        all_predictions = {}
        for prediction in json_predictions:
            # 检查 prediction 是否已经是字典
            if isinstance(prediction, dict):
                all_predictions.update(prediction)
            else:
                # 如果是 JSON 字符串，则解析它
                all_predictions.update(json.loads(prediction))
        
        return all_predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)