from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
from test_model import get_json_predictions_async  # 导入异步版本函数

app = FastAPI()

#同一时间只允许一个请求执行
predict_semaphore = asyncio.Semaphore(1)

class PredictionRequest(BaseModel):
    beatmap_ids: List[int]

@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Dict[str, float]]:
    """
    异步预测接口，使用信号量确保同一时间只有一个请求在执行

    参数:
        request: 包含beatmap ID列表的请求体

    返回:
        格式化的预测结果，结构为 {beatmap_id: {category: confidence}}
    """
    async with predict_semaphore:
        try:
            if not request.beatmap_ids:
                raise HTTPException(status_code=400, detail="至少需要一个有效的beatmap ID")

            # 转换为字符串列表（原函数需要）
            beatmap_ids = [str(id) for id in request.beatmap_ids]

            # 调用异步预测函数
            results = await get_json_predictions_async(["models"], beatmap_ids)

            # 合并每个 beatmap ID 的预测结果
            formatted_results = {}
            for beatmap_id, predictions in results.items():
                merged_prediction = {}
                for prediction in predictions:
                    merged_prediction.update(prediction)
                formatted_results[beatmap_id] = merged_prediction

            return formatted_results

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"无效的beatmap ID: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"预测出错: {str(e)}"
            )

@app.post("/predict_unlimited")
async def predict_unlimited(request: PredictionRequest) -> Dict[str, Dict[str, float]]:
    """
    功能与 /predict 完全相同，但没有速率限制
    """
    try:
        if not request.beatmap_ids:
            raise HTTPException(status_code=400, detail="至少需要一个有效的beatmap ID")

        beatmap_ids = [str(id) for id in request.beatmap_ids]
        results = await get_json_predictions_async(["models"], beatmap_ids)

        formatted_results = {}
        for beatmap_id, predictions in results.items():
            merged_prediction = {}
            for prediction in predictions:
                merged_prediction.update(prediction)
            formatted_results[beatmap_id] = merged_prediction

        return formatted_results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效的beatmap ID: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测出错: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7777,
    )