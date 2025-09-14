#!/usr/bin/env python3
"""
Простой скрипт обучения YOLOv8 для классификации грязных/чистых машин
Только обучение - никаких лишних функций
"""

import os
from ultralytics import YOLO

def main():
    # Настройки
    data_path = "../dataset"
    model_size = "s"  # yolov8n - самая быстрая
    epochs = 50       # меньше эпох для быстрого теста
    device = "1"      # GPU 0 (после CUDA_VISIBLE_DEVICES=1)
    
    print(f"🚀 Начинаем обучение YOLOv8{model_size}")
    print(f"📁 Данные: {data_path}")
    print(f"🎯 Эпохи: {epochs}")
    print(f"💻 GPU: {device}")
    
    # Проверяем данные
    if not os.path.exists(data_path):
        print("❌ Данные не найдены! Конвертируем...")
        os.system("cd ../dataset_raw && python convert_yolo_to_classification.py --input dataset_raw --output data_cars_converted")
    
    # Загружаем модель
    model = YOLO(f"yolov8{model_size}-cls.pt")
    
    # Обучаем
    results = model.train(
        data=data_path,
        epochs=epochs,
        device=device,
        project="runs/classify",
        name="dirty_car_simple",
        verbose=True
    )
    
    # Сохраняем лучшую модель
    best_model_path = "runs/classify/dirty_car_simple/weights/best.pt"
    
    print(f"✅ Обучение завершено!")
    print(f"📦 Модель сохранена: {best_model_path}")
    
    # Копируем модель в удобное место
    os.makedirs("trained_models", exist_ok=True)
    os.system(f"cp {best_model_path} trained_models/dirty_car_yolo.pt")
    print(f"📋 Модель скопирована: trained_models/dirty_car_yolo.pt")

if __name__ == "__main__":
    main()
