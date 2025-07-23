from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
from test_model import get_json_predictions_async  # 导入异步版本函数

app = FastAPI()


class PredictionRequest(BaseModel):
    beatmap_ids: List[int]


@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Dict[str, float]]:
    """
    异步预测接口

    参数:
        request: 包含beatmap ID列表的请求体

    返回:
        格式化的预测结果，结构为 {beatmap_id: {category: confidence}}
    """
    try:
        if not request.beatmap_ids:
            raise HTTPException(status_code=400, detail="至少需要一个有效的beatmap ID")

        # 转换为字符串列表（原函数需要）
        beatmap_ids = [str(id) for id in request.beatmap_ids]

        # 调用异步预测函数
        results = await get_json_predictions_async(["models"], beatmap_ids)

        # 处理并格式化结果
        formatted_results = {}
        for beatmap_id, predictions in results.items():
            if predictions is None:
                continue  # 跳过处理失败的beatmap

            # 合并所有模型的预测结果
            merged_prediction = {}
            for prediction in predictions:
                merged_prediction.update(prediction)

            # 只保留置信度大于5%的结果
            filtered_prediction = {
                k: round(v, 4)  # 保留4位小数
                for k, v in merged_prediction.items()
                if v > 0.05
            }

            if filtered_prediction:  # 只添加有有效预测的结果
                formatted_results[beatmap_id] = filtered_prediction

        return formatted_results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效的beatmap ID: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"预测出错: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=7777,
        workers=2,
        limit_concurrency=100,
        timeout_keep_alive=60
    )