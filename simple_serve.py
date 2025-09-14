#!/usr/bin/env python3
"""
Простой FastAPI сервер для YOLO модели классификации
Получает фото -> отдает ответ (clean/dirty)
"""

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Модели ответов
class PredictionResponse(BaseModel):
    label: str          # "clean" или "dirty"
    confidence: float   # уверенность 0-1
    processing_time: float  # время обработки в мс

class DamageResponse(BaseModel):
    damage_results: dict    # результаты детекции повреждений
    scratch_results: dict   # результаты детекции царапин
    processing_time: float  # время обработки в мс
    combined_detections: list  # все найденные дефекты

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    damage_model_loaded: bool
    scratch_model_loaded: bool

# Глобальные модели
model = None  # модель clean/dirty
damage_model = None  # модель повреждений
scratch_model = None  # модель царапин

def load_model():
    """Загружаем обученную YOLO модель для clean/dirty"""
    global model
    
    # Пути к модели (в порядке приоритета)
    model_paths = [
        "trained_models/clean_dirty_best.pt",  # основная модель clean/dirty
        "trained_models/dirty_car_yolo.pt",
        "runs/classify/dirty_car_simple2/weights/best.pt",
        "runs/classify/dirty_car_simple/weights/best.pt",
        "artifacts/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"🔥 Загружаем clean/dirty модель: {path}")
            model = YOLO(path)
            return True
    
    print("❌ Clean/dirty модель не найдена!")
    return False

def load_damage_model():
    """Загружаем модель для детекции повреждений"""
    global damage_model
    
    # Пути к модели повреждений
    damage_paths = [
        "trained_models/damage_best.pt",  # модель повреждений
        "trained_models/best.pt",
        "damage_model/best.pt",
        "models/damage_best.pt"
    ]
    
    for path in damage_paths:
        if os.path.exists(path):
            print(f"🔥 Загружаем damage модель: {path}")
            damage_model = YOLO(path)
            return True
    
    print("❌ Damage модель не найдена!")
    return False

def load_scratch_model():
    """Загружаем модель для детекции царапин"""
    global scratch_model
    
    # Пути к модели царапин
    scratch_paths = [
        "trained_models/scratch_best.pt",  # модель царапин
        "trained_models/scratch_model.pt",
        "trained_models/scratch.pt",
        "models/scratch_best.pt",
        "scratch_model/best.pt"
    ]
    
    for path in scratch_paths:
        if os.path.exists(path):
            print(f"🔥 Загружаем scratch модель: {path}")
            scratch_model = YOLO(path)
            return True
    
    print("❌ Scratch модель не найдена!")
    return False

def predict_image(image_bytes):
    """Предсказание для изображения (clean/dirty)"""
    start_time = time.time()
    
    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    
    # Предсказание YOLO
    if model is None:
        raise ValueError("Модель не загружена")
    results = model(image, verbose=False)
    
    # Извлекаем результат
    probs = results[0].probs
    class_id = probs.top1  # ID класса с максимальной вероятностью
    confidence = float(probs.top1conf)  # Уверенность
    
    # Определяем метку (предполагаем: 0=clean, 1=dirty)
    label = "clean" if class_id == 0 else "dirty"
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "label": label,
        "confidence": confidence,
        "processing_time": processing_time
    }

def predict_damage_and_scratch(image_bytes):
    """Предсказание повреждений и царапин автомобиля"""
    start_time = time.time()
    
    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    
    # Классы повреждений
    damage_classes = {
        0: 'RunningBoard-Dent',  # вмятина на подножке
        1: 'bonnet-dent',        # вмятина на капоте
        2: 'dent',               # общая вмятина
        3: 'doorouter-dent',     # вмятина на внешней двери
        4: 'fender-dent',        # вмятина на крыле
        5: 'front-bumper-dent',  # вмятина на переднем бампере
        6: 'quaterpanel-dent',   # вмятина на задней панели
        7: 'rear-bumper-dent',   # вмятина на заднем бампере
        8: 'roof-dent'           # вмятина на крыше
    }
    
    # Результаты повреждений
    damage_results = {"detections": [], "best_detection": None}
    if damage_model is not None:
        try:
            damage_pred = damage_model(image, verbose=False)
            if hasattr(damage_pred[0], 'boxes') and damage_pred[0].boxes is not None:
                boxes = damage_pred[0].boxes
                for i in range(len(boxes.cls)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    damage_type = damage_classes.get(class_id, f"unknown_{class_id}")
                    
                    detection = {
                        "type": damage_type,
                        "confidence": confidence,
                        "bbox": boxes.xyxy[i].tolist() if hasattr(boxes, 'xyxy') else None
                    }
                    damage_results["detections"].append(detection)
                
                if damage_results["detections"]:
                    damage_results["best_detection"] = max(damage_results["detections"], key=lambda x: x['confidence'])
        except Exception as e:
            print(f"Ошибка damage модели: {e}")
    
    # Результаты царапин
    scratch_results = {"detections": [], "best_detection": None}
    if scratch_model is not None:
        try:
            scratch_pred = scratch_model(image, verbose=False)
            if hasattr(scratch_pred[0], 'boxes') and scratch_pred[0].boxes is not None:
                boxes = scratch_pred[0].boxes
                for i in range(len(boxes.cls)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    scratch_type = f"scratch_{class_id}"  # можешь добавить свои классы царапин
                    
                    detection = {
                        "type": scratch_type,
                        "confidence": confidence,
                        "bbox": boxes.xyxy[i].tolist() if hasattr(boxes, 'xyxy') else None
                    }
                    scratch_results["detections"].append(detection)
                
                if scratch_results["detections"]:
                    scratch_results["best_detection"] = max(scratch_results["detections"], key=lambda x: x['confidence'])
        except Exception as e:
            print(f"Ошибка scratch модели: {e}")
    
    # Объединяем все детекции
    combined_detections = []
    combined_detections.extend(damage_results["detections"])
    combined_detections.extend(scratch_results["detections"])
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "damage_results": damage_results,
        "scratch_results": scratch_results,
        "processing_time": processing_time,
        "combined_detections": combined_detections
    }

# FastAPI приложение
app = FastAPI(
    title="DirtyCar YOLO API",
    description="Простой API для классификации чистых/грязных машин",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Загружаем модели при запуске"""
    clean_dirty_loaded = load_model()
    damage_loaded = load_damage_model()
    scratch_loaded = load_scratch_model()
    
    if not clean_dirty_loaded:
        print("⚠️ Clean/dirty модель не загружена")
    if not damage_loaded:
        print("⚠️ Damage модель не загружена")
    if not scratch_loaded:
        print("⚠️ Scratch модель не загружена")
    
    if not clean_dirty_loaded and not damage_loaded and not scratch_loaded:
        raise Exception("Ни одна модель не загружена!")

@app.get("/healthz", response_model=HealthResponse)
async def health():
    """Проверка здоровья API"""
    return HealthResponse(
        status="ok" if (model is not None or damage_model is not None or scratch_model is not None) else "error",
        model_loaded=model is not None,
        damage_model_loaded=damage_model is not None,
        scratch_model_loaded=scratch_model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Основной эндпойнт для предсказания
    Загружаешь фото -> получаешь ответ
    """
    # Проверяем тип файла
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем файл
        contents = await file.read()
        
        # Предсказание
        result = predict_image(contents)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.post("/predict/damage", response_model=DamageResponse)
async def predict_car_damage(file: UploadFile = File(...)):
    """
    Эндпойнт для детекции повреждений автомобиля
    Загружаешь фото -> получаешь тип повреждения
    """
    # Проверяем тип файла
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем файл
        contents = await file.read()
        
        # Предсказание повреждений и царапин
        result = predict_damage_and_scratch(contents)
        
        return DamageResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка детекции повреждений: {str(e)}")

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "🚗 DirtyCar YOLO API",
        "endpoints": {
            "predict": "/predict (POST с файлом) - clean/dirty",
            "predict_damage": "/predict/damage (POST с файлом) - повреждения",
            "health": "/healthz",
            "docs": "/docs"
        },
        "usage": {
            "clean_dirty": "curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict",
            "damage": "curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict/damage"
        }
    }

if __name__ == "__main__":
    # Запуск сервера
    port = int(os.getenv('API_PORT', '7439'))
    
    uvicorn.run(
        "simple_serve:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
