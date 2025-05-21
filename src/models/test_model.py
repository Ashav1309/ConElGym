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

def test_model(model_path, test_data_path, model_type=None, batch_size=1):
    """
    Тестирование модели на тестовых данных
    Args:
        model_path: путь к сохраненной модели
        test_data_path: путь к тестовым данным
        model_type: тип модели ('v3' или 'v4'). Если None, определяется из пути модели
        batch_size: размер батча
    """
    try:
        print("\n[DEBUG] ===== Начало тестирования модели =====")
        
        # Проверяем существование директорий и файлов
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Директория с тестовыми данными не найдена: {test_data_path}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
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
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)['model']
                print(f"[DEBUG] ✓ Модель успешно загружена из pickle")
            else:
                model = load_model(model_path)
                print(f"[DEBUG] ✓ Модель успешно загружена из h5")
            print(f"[DEBUG] Архитектура модели: {model.summary()}")
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке модели: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise

        # Загрузка данных
        print("\n[DEBUG] Загрузка тестовых данных...")
        try:
            loader = VideoDataLoader(test_data_path)
            generator = loader.load_data(
                Config.SEQUENCE_LENGTH,
                batch_size,
                target_size=Config.INPUT_SIZE,
                one_hot=True,
                infinite_loop=False
            )
            print("[DEBUG] ✓ Тестовые данные загружены")
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise

        # Тестирование
        print("\n[DEBUG] Начало тестирования...")
        video_idx = 0
        results = []
        all_y_true = []
        all_y_pred = []
        
        try:
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
                    # Для метрик: flatten по всем кадрам, класс "action" (индекс 1)
                    all_y_true.extend(np.argmax(true, axis=-1).flatten())
                    all_y_pred.extend(np.argmax(pred, axis=-1).flatten())
                    video_idx += 1
        except Exception as e:
            print(f"[ERROR] Ошибка при тестировании: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Считаем метрики по классу "action"
        print("\n[DEBUG] Расчёт метрик по классу 'action'...")
        f1 = f1_score(all_y_true, all_y_pred, pos_label=1, average='binary')
        precision = precision_score(all_y_true, all_y_pred, pos_label=1, average='binary')
        recall = recall_score(all_y_true, all_y_pred, pos_label=1, average='binary')
        print(f"F1-score (action): {f1:.4f}")
        print(f"Precision (action): {precision:.4f}")
        print(f"Recall (action): {recall:.4f}")
        
        # Сохраняем результаты
        print("\n[DEBUG] Сохранение результатов...")
        try:
            results_dir = os.path.join(Config.MODEL_SAVE_PATH, model_type, 'test_results')
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, 'test_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'model_path': model_path,
                    'f1_action': f1,
                    'precision_action': precision,
                    'recall_action': recall,
                    'results': results
                }, f, indent=4)
            
            print(f"[DEBUG] ✓ Результаты сохранены в {results_file}")
        except Exception as e:
            print(f"[ERROR] Ошибка при сохранении результатов: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        print("[DEBUG] ===== Тестирование завершено =====\n")
        
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при тестировании: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

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
            test_data_path=Config.TEST_DATA_PATH,
            model_type='v3'
        )
        
        print("\n[DEBUG] Тестирование MobileNetV4...")
        test_model(
            model_path=os.path.join(Config.MODEL_SAVE_PATH, 'v4', 'best_model.h5'),
            test_data_path=Config.TEST_DATA_PATH,
            model_type='v4'
        )
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при запуске тестирования: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1) 