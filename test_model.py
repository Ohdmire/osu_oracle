import gdown
import keras
import numpy as np
import os
import pickle
import sys
import tempfile
import time
import zipfile
import concurrent.futures
import threading
import aiohttp
import asyncio

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 下载模型文件（如果不存在）
if not os.path.exists('models'):
    url = 'https://drive.google.com/uc?id=14zLtVPcBDyLP-Rlj-2b6-mPIqGq-JY9J'
    output = 'models.zip'
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile("models.zip", 'r') as zip_ref:
        zip_ref.extractall("models")

async def download_beatmap_async(beatmap_id):
    url = f"https://osu.direct/api/osu/{beatmap_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    print(f"Error fetching .osu file for beatmap ID {beatmap_id}: {response.status}")
                    return None
                return await response.read()
    except Exception as e:
        print(f"Download error for beatmap {beatmap_id}: {str(e)}")
        return None


async def process_beatmap_async(beatmap_id, models, max_slider_length, max_time_diff, label_encoder_path):
    osu_file_content = await download_beatmap_async(beatmap_id)
    if osu_file_content is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(osu_file_content)
        temp_file_path = temp_file.name

    beatmap_data = parse_osu_file(temp_file_path, max_slider_length, print_info=True)
    os.unlink(temp_file_path)

    if beatmap_data is None:
        print(f"Error processing beatmap ID {beatmap_id}: Unsupported mode or invalid file")
        return None

    json_predictions = []
    for model in models:
        input_shape = model[0].input_shape
        if isinstance(input_shape, list):
            max_sen = input_shape[0][1]
        else:
            max_sen = input_shape[1]
        prediction_dict = get_predictions_as_dict(beatmap_data, model, max_sen, max_slider_length, max_time_diff, label_encoder_path)
        json_predictions.append(prediction_dict)

    return json_predictions

def parse_osu_file(file_path, max_slider_length = 1, max_time_diff = 1, print_info = False):
    data = {
        'beatmap_id': None,
        'hp_drain': None,
        'circle_size': None,
        'od': None,
        'ar': None,
        'slider_multiplier': None,
        'slider_tick': None,
        'hit_objects': [],
        'label': None,
    }

    parent_folder = os.path.dirname(file_path)
    data['label'] = os.path.basename(parent_folder)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue

            if section == 'General':
                if line.startswith('Mode:'):
                    mode = int(line.split(':')[1].strip())
                    if mode != 0:
                        print(f"Unsupported mode: {mode}. Only osu!standard mode (Mode: 0) is supported.")
                        return None


            if section == 'Metadata':
                key, value = line.split(':', maxsplit=1)
                if key == 'Title' and print_info:
                    print("Title: " + value, end = ' ')
                if key == 'Artist' and print_info:
                    print("by " + value)
                if key == 'Creator' and print_info:
                    print("Mapper: " + value)
                if key == 'Version' and print_info:
                    print("Diffuculty: " + value)
                if key == 'BeatmapID':
                    data['beatmap_id'] = int(value)
                    print("ID: " + value)
                    print("----------------------------------------------------")
                    if data['beatmap_id'] is None:
                        return None
            elif section == 'Difficulty':
                key, value = line.split(':', maxsplit=1)
                value = float(value)
                if key == 'HPDrainRate':
                    data['hp_drain'] = value
                elif key == 'CircleSize':
                    data['circle_size'] = value
                elif key == 'OverallDifficulty':
                    data['od'] = value
                elif key == 'ApproachRate':
                    data['ar'] = value
                elif key == 'SliderMultiplier':
                    data['slider_multiplier'] = value
                elif key == 'SliderTickRate':
                    data['slider_tick'] = value
            elif section == 'HitObjects':  # Move this line one level back
                    obj_data = line.split(',')
                    hit_object_type = int(obj_data[3])

                    hit_circle_flag = 0b1
                    slider_flag = 0b10

                    if hit_object_type & hit_circle_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': float(0), # slider len
                        }
                        data['hit_objects'].append(hit_object)
                    elif hit_object_type & slider_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': min(500, float(obj_data[7])), # slider len 
                        }
                        data['hit_objects'].append(hit_object)
                        
    # Normalize the coordinates
    max_x, max_y = 512, 384
    for obj in data['hit_objects']:
        obj['x_norm'] = obj['x'] / max_x
        obj['y_norm'] = obj['y'] / max_y

    # Compute the time differences
    if data['hit_objects']:  # Add this condition
        for i, obj in enumerate(data['hit_objects'][1:], start=1):
            obj['time_diff'] = obj['time'] - data['hit_objects'][i - 1]['time']
        data['hit_objects'][0]['time_diff'] = 0

    vectors = []
    for i, obj in enumerate(data['hit_objects'][1:], start=1):
        prev_obj = data['hit_objects'][i - 1]
        x_diff = round(obj['x_norm'] - prev_obj['x_norm'], 4)
        y_diff = round(obj['y_norm'] - prev_obj['y_norm'], 4)
        time_diff = round(obj['time_diff'], 4)
        length = round(obj['length'], 4)
        vectors.append((x_diff, y_diff, time_diff / max_time_diff, length / max_slider_length))

    data['vectors'] = vectors
    return data

def get_predictions_as_dict(beatmap_data, bagged_models, max_sequence_length, max_slider_length, max_time_diff, label_encoder_path):
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)

    # Use the average prediction of the bagged models
    y_preds = []
    for model in bagged_models:
        y_pred = model.predict(beatmap_vectors_padded, verbose=0)
        y_preds.append(y_pred)

    y_preds_mean = np.mean(y_preds, axis=0)

    # Determine the predicted category and confidence for each category
    categories = label_encoder.inverse_transform(range(len(y_preds_mean[0])))
    confidences = y_preds_mean[0]

    # Create a list of (category, confidence) tuples
    predictions = list(zip(categories, confidences))

    # Sort the predictions by confidence in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Prepare the JSON object for predictions with confidences greater than 5%
    json_predictions = {category: float(confidence) for category, confidence in sorted_predictions if confidence > 0.05}

    # Return the JSON object
    return json_predictions


async def get_json_predictions_async(folders, beatmap_ids):
    """获取预测结果的函数

    参数:
        folders: 包含模型的文件夹列表
        beatmap_ids: 要预测的beatmap ID列表

    返回:
        包含预测结果的字典，格式为 {beatmap_id: [prediction1, prediction2, ...]}
    """
    # 加载模型
    models = []
    start = time.time()
    for folder in folders:
        model_folder = folder + "/"
        model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
        model = [load_model(model_path, compile=False) for model_path in model_paths]
        models.append(model)
    end = time.time()
    max_slider_length = 500.0
    max_time_diff = 1000
    label_encoder_path = model_folder + "label_encoder.pkl"
    print(f"模型加载时间: {round(end - start, 2)}秒")

    # 异步处理所有beatmap
    tasks = []
    for beatmap_id in beatmap_ids:
        task = asyncio.create_task(
            process_beatmap_async(beatmap_id, models, max_slider_length, max_time_diff, label_encoder_path)
        )
        tasks.append(task)

    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    final_results = {}
    for beatmap_id, result in zip(beatmap_ids, results):
        if isinstance(result, Exception):
            print(f"处理beatmap {beatmap_id}时出错: {str(result)}")
            final_results[beatmap_id] = None
        elif result is not None:
            final_results[beatmap_id] = result

    return final_results

# def main(beatmap_ids):
#     folders = ["models"]
#     results = get_json_predictions(folders, beatmap_ids)
#     for beatmap_id, predictions in results.items():
#         print(f"Beatmap ID: {beatmap_id}")
#         for i, prediction in enumerate(predictions):
#             print(f"Model {i+1} predictions:")
#             print(prediction)
#         print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py [beatmap_id1] [beatmap_id2] ...")
    else:
        beatmap_ids = sys.argv[1:]
        results = get_json_predictions(["models"], beatmap_ids)
        for beatmap_id, predictions in results.items():
            print(f"Beatmap ID: {beatmap_id}")
            for i, prediction in enumerate(predictions):
                print(f"Model {i+1} predictions:")
                print(prediction)
            print()
