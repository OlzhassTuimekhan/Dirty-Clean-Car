#!/bin/bash
# Запуск API сервера для DirtyCar

echo "🚀 Запускаем DirtyCar API сервер..."

# Проверяем модель
if [ ! -f "trained_models/dirty_car_yolo.pt" ] && [ ! -d "runs/classify" ]; then
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
