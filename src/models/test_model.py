import os
import numpy as np
from tensorflow.keras.models import load_model
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from src.models.inference_utils import get_element_intervals
import json
import sys
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from src.models.metrics import calculate_metrics
import tensorflow as tf

def load_model(model_path=None, model_type=None):
    """
    Загрузка модели для тестирования
    
    Args:
        model_path: путь к файлу модели
        model_type: тип модели ('v3'). Если None, определяется из пути модели
    """
    if model_path is None:
        model_path = os.path.join(Config.MODEL_SAVE_PATH, 'v3', 'best_model.h5')
    
    if model_type is None:
        if 'v3' in model_path:
            model_type = 'v3'
        else:
            raise ValueError("Не удалось определить тип модели из пути")
    
    print(f"\n[DEBUG] Загрузка модели типа {model_type} из {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("[DEBUG] Модель успешно загружена")
        return model
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке модели: {str(e)}")
        raise

def test_model(model_path=None, model_type=None):
    """
    Тестирование модели на валидационном наборе данных
    
    Args:
        model_path: путь к файлу модели
        model_type: тип модели ('v3'). Если None, определяется из пути модели
    """
    print("\n[DEBUG] Тестирование MobileNetV3...")
    model = load_model(
        model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v3', 'best_model.h5'),
        model_type='v3'
    )

def evaluate_model(model, test_data):
    """
    Оценка модели на тестовых данных
    """
    all_y_true = []
    all_y_pred = []
    
    for X, y in test_data:
        y_pred = model.predict(X)
        all_y_true.extend(y.numpy())
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Рассчитываем метрики
    metrics = calculate_metrics(all_y_true, all_y_pred)
    
    print("\nРезультаты оценки модели:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (action): {metrics['precision_action']:.4f}")
    print(f"Recall (action): {metrics['recall_action']:.4f}")
    print(f"F1-Score (action): {metrics['f1_action']:.4f}")
    
    return metrics

if __name__ == "__main__":
    try:
        # Тестируем обе модели
        print("[DEBUG] ===== Запуск тестирования моделей =====")
        
        print("\n[DEBUG] Тестирование MobileNetV3...")
        test_model(
            model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v3', 'best_model.h5'),
            model_type='v3'
        )
        
        print("\n[DEBUG] Тестирование MobileNetV4...")
        test_model(
            model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v4', 'best_model.h5'),
            model_type='v4'
        )
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при запуске тестирования: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1) 