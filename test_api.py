#!/usr/bin/env python3
"""
Тест API - загружает фото и получает предсказание
"""

import requests
import sys

def test_api(image_path, api_url="http://localhost:7439"):
    """Тестируем API с изображением"""
    
    print(f"🧪 Тестируем API: {api_url}")
    print(f"📸 Изображение: {image_path}")
    
    # Проверяем здоровье API
    try:
        health = requests.get(f"{api_url}/healthz")
        print(f"✅ Health check: {health.json()}")
    except:
        print("❌ API недоступен!")
        return
    
    # Отправляем изображение
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Результат:")
            print(f"  Метка: {result['label']}")
            print(f"  Уверенность: {result['confidence']:.3f}")
            print(f"  Время: {result['processing_time']:.1f}ms")
        else:
            print(f"❌ Ошибка: {response.status_code} - {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Файл не найден: {image_path}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python test_api.py <путь_к_изображению>")
        sys.exit(1)
    
    test_api(sys.argv[1])
