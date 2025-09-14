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

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Глобальная модель
model = None

def load_model():
    """Загружаем обученную YOLO модель"""
    global model
    
    # Пути к модели
    model_paths = [
        "trained_models/dirty_car_yolo.pt",
        "runs/classify/dirty_car_simple/weights/best.pt",
        "artifacts/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"🔥 Загружаем модель: {path}")
            model = YOLO(path)
            return True
    
    print("❌ Модель не найдена!")
    return False

def predict_image(image_bytes):
    """Предсказание для изображения"""
    start_time = time.time()
    
    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    
    # Предсказание YOLO
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
    """Загружаем модель при запуске"""
    if not load_model():
        raise Exception("Не удалось загрузить модель!")

@app.get("/healthz", response_model=HealthResponse)
async def health():
    """Проверка здоровья API"""
    return HealthResponse(
        status="ok" if model is not None else "error",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Основной эндпойнт для предсказания
    Загружаешь фото -> получаешь ответ
    """
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем файл
        contents = await file.read()
        
        # Предсказание
        result = predict_image(contents)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "🚗 DirtyCar YOLO API",
        "endpoints": {
            "predict": "/predict (POST с файлом)",
            "health": "/healthz",
            "docs": "/docs"
        },
        "usage": "curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict"
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
