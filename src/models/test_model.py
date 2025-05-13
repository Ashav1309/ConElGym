import os
import numpy as np
from tensorflow.keras.models import load_model
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from src.models.inference_utils import get_element_intervals
import json

def test_model(model_path, test_data_path, model_type=None, batch_size=1):
    """
    Тестирование модели на тестовых данных
    Args:
        model_path: путь к сохраненной модели
        test_data_path: путь к тестовым данным
        model_type: тип модели ('v3' или 'v4'). Если None, определяется из пути модели
        batch_size: размер батча
    """
    print("\n[DEBUG] ===== Начало тестирования модели =====")
    
    # Определяем тип модели из пути, если не указан
    if model_type is None:
        if 'v3' in model_path:
            model_type = 'v3'
        elif 'v4' in model_path:
            model_type = 'v4'
        else:
            model_type = Config.MODEL_TYPE
    
    print(f"[DEBUG] Тип модели: {model_type}")
    print(f"[DEBUG] Путь к модели: {model_path}")
    print(f"[DEBUG] Путь к тестовым данным: {test_data_path}")
    print(f"[DEBUG] Размер батча: {batch_size}")
    
    # Загрузка модели
    print("\n[DEBUG] Загрузка модели...")
    model = load_model(model_path)
    print(f"[DEBUG] ✓ Модель успешно загружена")
    print(f"[DEBUG] Архитектура модели: {model.summary()}")

    # Загрузка данных
    print("\n[DEBUG] Загрузка тестовых данных...")
    loader = VideoDataLoader(test_data_path)
    generator = loader.load_data(
        Config.SEQUENCE_LENGTH,
        batch_size,
        target_size=Config.INPUT_SIZE,
        one_hot=True,
        infinite_loop=False
    )
    print("[DEBUG] ✓ Тестовые данные загружены")

    # Тестирование
    print("\n[DEBUG] Начало тестирования...")
    video_idx = 0
    results = []
    
    for X_batch, y_true in generator:
        print(f"\n[DEBUG] Обработка батча {video_idx // batch_size + 1}")
        preds = model.predict(X_batch)
        for i, (pred, true) in enumerate(zip(preds, y_true)):
            result = get_element_intervals(pred)
            print(f"[DEBUG] Видео {video_idx}:")
            print(f"  [DEBUG] Предсказанные интервалы: {result}")
            print(f"  [DEBUG] Истинные метки: {true}")
            results.append({
                'video_idx': video_idx,
                'predicted_intervals': result,
                'true_labels': true.tolist()
            })
            video_idx += 1
    
    # Сохраняем результаты
    print("\n[DEBUG] Сохранение результатов...")
    results_dir = os.path.join(Config.MODEL_SAVE_PATH, model_type, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'model_path': model_path,
            'results': results
        }, f, indent=4)
    
    print(f"[DEBUG] ✓ Результаты сохранены в {results_file}")
    print("[DEBUG] ===== Тестирование завершено =====\n")

if __name__ == "__main__":
    # Тестируем обе модели
    print("[DEBUG] ===== Запуск тестирования моделей =====")
    
    print("\n[DEBUG] Тестирование MobileNetV3...")
    test_model(
        model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v3', 'best_model.h5'),
        test_data_path=Config.TEST_DATA_PATH,
        model_type='v3'
    )
    
    print("\n[DEBUG] Тестирование MobileNetV4...")
    test_model(
        model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v4', 'best_model.h5'),
        test_data_path=Config.TEST_DATA_PATH,
        model_type='v4'
    ) 