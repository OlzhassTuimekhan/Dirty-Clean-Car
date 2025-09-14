#!/bin/bash
# Запуск API сервера для DirtyCar

echo "🚀 Запускаем DirtyCar API сервер..."

# Проверяем модель (в порядке приоритета)
if [ -f "runs/classify/dirty_car_simple2/weights/best.pt" ]; then
    echo "✅ Найдена новая модель: runs/classify/dirty_car_simple2/weights/best.pt"
elif [ -f "runs/classify/dirty_car_simple/weights/best.pt" ]; then
    echo "✅ Найдена модель: runs/classify/dirty_car_simple/weights/best.pt"
elif [ -f "trained_models/dirty_car_yolo.pt" ]; then
    echo "✅ Найдена модель: trained_models/dirty_car_yolo.pt"
else
    echo "❌ Модель не найдена! Сначала обучите модель:"
    echo "   python simple_train.py"
    exit 1
fi

# Устанавливаем зависимости если нужно
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "📦 Устанавливаем зависимости..."
    pip install -r requirements_server.txt
fi

# Запускаем сервер
export API_PORT=7439
echo "🌐 API будет доступен на: http://localhost:7439"
echo "📚 Документация: http://localhost:7439/docs"
echo "❤️  Здоровье: http://localhost:7439/healthz"
echo ""
echo "Для тестирования:"
echo "curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict"
echo ""

python simple_serve.py
