# 🚗 DirtyCar Classification Project

**Автоматическая классификация автомобилей на чистые и грязные** с использованием YOLOv8 и FastAPI.

## 📊 Результаты
- **Точность:** 99.7% на валидации
- **Классы:** Clean (64,594 изображений) / Dirty (2,190 изображений)  
- **Архитектура:** YOLOv8s-cls (5.08M параметров)
- **Скорость:** ~200 FPS на RTX 3090

## 🚀 Быстрый старт

### 1. Обучение модели
```bash
python simple_train.py
```
- Автоматически найдет данные в `dataset/clean/` и `dataset/dirty/`
- Обучит YOLOv8s на 50 эпох
- Сохранит модель в `runs/classify/dirty_car_simple2/weights/best.pt`

### 2. Запуск API сервера
```bash
# Автоматический запуск (рекомендуется)
python setup_server.py

# Или через bash
bash start_server.sh

# Или вручную
pip install -r requirements_server.txt
python simple_serve.py
```

### 3. Тестирование API
```bash
# Через curl
curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict

# Через Python
python test_api.py
```

## 🌐 API Endpoints

### POST `/predict`
Загрузка изображения для классификации
```bash
curl -X POST -F 'file=@car.jpg' http://localhost:7439/predict
```

**Ответ:**
```json
{
  "label": "clean",
  "confidence": 0.987,
  "processing_time": 15.2
}
```

### GET `/healthz`
Проверка состояния сервера
```bash
curl http://localhost:7439/healthz
```

### GET `/docs`
Интерактивная документация Swagger UI
```
http://localhost:7439/docs
```

## Business Logic

The model uses a threshold-based approach:
- If `p_clean ≥ T_clean` → classify as **clean**
- Otherwise → classify as **dirty**

The threshold `T_clean` is optimized on validation data to achieve target precision ≥ 0.95 for the **clean** class.

## Features

- **Multi-GPU Training**: DDP support for 2×RTX 3090 with mixed precision (AMP)
- **Class Imbalance Handling**: WeightedRandomSampler + oversampling options
- **Advanced Models**: timm integration (ResNet-50, EfficientNet, ConvNeXt)
- **Loss Functions**: CrossEntropy with class weights + Focal Loss option
- **Threshold Optimization**: Precision-recall based threshold selection
- **Production Ready**: ONNX export + FastAPI service + Docker deployment
- **Comprehensive Metrics**: TensorBoard logging + confusion matrices + CSV reports

## Quick Start

### Environment Setup

Organize your data as follows:
```
data_cars/
├── train/
│   ├── clean/
│   └── dirty/
├── val/
│   ├── clean/
│   └── dirty/
└── test/
    ├── clean/
    └── dirty/
```

## Training Pipeline

```bash
# Complete pipeline: train → eval → threshold → export → serve
bash scripts/run_all.sh

# Or step by step:
# 1. Training (2×GPU DDP)
bash scripts/launch_ddp.sh

# 2. Evaluation
python src/eval.py --checkpoint artifacts/best.pt --data-root data_cars

# 3. Threshold selection (target precision ≥ 0.95 for clean class)
python src/threshold_select.py --probs-source val --target-precision-clean 0.95 \
    --checkpoint artifacts/best.pt --out artifacts/threshold.json

# 4. ONNX export
python src/export_onnx.py --checkpoint artifacts/best.pt --imgsz 256 --half \
    --out artifacts/best.onnx

# 5. Start API service
MODEL_PATH=artifacts/best.onnx IMGSZ=256 \
    python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run with GPU support
docker-compose up --build

# Or manually:
docker build -t dirtycar:latest -f docker/Dockerfile .
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/artifacts:/app/artifacts \
    dirtycar:latest
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/healthz
```

### File Upload Prediction
```bash
curl -X POST "http://localhost:8000/predict/file" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@car_image.jpg"
```

### URL Prediction
```bash
curl -X POST "http://localhost:8000/predict/url" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com/car.jpg"}'
```

### Response Format
```json
{
    "label": "clean",
    "p_clean": 0.87,
    "threshold": 0.83
}
```

## Configuration

Main training configuration in `configs/train.yaml`:

```yaml
data_root: "./data_cars"
img_size: 256
batch_size: 192
epochs: 40
lr: 3e-4
weight_decay: 1e-4
sampler: "weighted"  # "weighted" | "oversample" | "none"
oversample_ratio: 1.0
class_weights: "auto"  # [w_clean, w_dirty] | "auto"
loss: "ce"  # "ce" | "focal"
model: "resnet50"  # "resnet50" | "efficientnet_b0" | "convnext_tiny"
amp: true
ddp: true
num_workers: 8
```

## Project Structure

```
dirtycar/
├── README.md
├── requirements.txt
├── env.example
├── configs/
│   └── train.yaml
├── src/
│   ├── data_module.py
│   ├── losses.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   ├── export_onnx.py
│   ├── threshold_select.py
│   ├── serve.py
│   ├── utils_vis.py
│   └── infer_batch.py
├── scripts/
│   ├── run_all.sh
│   └── launch_ddp.sh
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data_cars/
│   ├── train/
│   ├── val/
│   └── test/
└── artifacts/
    ├── best.pt
    ├── best.onnx
    ├── threshold.json
    └── reports/
```

## Performance Optimization

- **Multi-GPU**: Utilizes 2×RTX 3090 with DistributedDataParallel
- **Mixed Precision**: AMP for ~2x speedup with minimal accuracy loss
- **Class Balancing**: Handles severe imbalance (dirty ≈ 2.3k, clean ≈ 69k)
- **ONNX Inference**: ~200 FPS on RTX 3090 with FP16 optimization
- **Batch Processing**: Efficient batch inference utilities

## Environment Variables

- `MODEL_PATH`: Path to ONNX model (default: `./artifacts/best.onnx`)
- `IMGSZ`: Input image size (default: `256`)
- `THRESH`: Manual threshold override
- `T_LOW`, `T_HIGH`: Optional uncertainty zone thresholds

## License

MIT License
