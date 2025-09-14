#!/usr/bin/env python3
"""
Автоматическая настройка и запуск API сервера
"""

import os
import sys
import subprocess

def install_dependencies():
    """Устанавливает минимальные зависимости для сервера"""
    print("📦 Устанавливаем зависимости для API сервера...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ultralytics>=8.0.0",
            "fastapi==0.111.0", 
            "uvicorn[standard]==0.30.0",
            "python-multipart==0.0.9",
            "numpy",
            "pillow",
            "opencv-python",
            "requests"
        ])
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка установки зависимостей")
        return False

def check_model():
    """Проверяет наличие обученной модели"""
    model_paths = [
        "runs/classify/dirty_car_simple2/weights/best.pt",  # новая обученная модель
        "runs/classify/dirty_car_simple/weights/best.pt",
        "trained_models/dirty_car_yolo.pt",
        "artifacts/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Найдена модель: {path}")
            return True
    
    print("❌ Модель не найдена!")
    print("Сначала обучите модель: python simple_train.py")
    return False

def start_server():
    """Запускает API сервер"""
    print("🚀 Запускаем API сервер на порту 7439...")
    print("🌐 API: http://localhost:7439")
    print("📚 Docs: http://localhost:7439/docs")
    print("❤️  Health: http://localhost:7439/healthz")
    print("")
    
    os.environ['API_PORT'] = '7439'
    
    try:
        subprocess.run([sys.executable, "simple_serve.py"])
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")

def main():
    print("🔧 Настройка DirtyCar API сервера...")
    
    # Проверяем модель
    if not check_model():
        return
    
    # Устанавливаем зависимости
    try:
        import ultralytics
        import fastapi
        print("✅ Зависимости уже установлены")
    except ImportError:
        if not install_dependencies():
            return
    
    # Запускаем сервер
    start_server()

if __name__ == "__main__":
    main()
