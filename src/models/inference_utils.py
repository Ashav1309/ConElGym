import numpy as np

def get_element_intervals(pred, threshold=0.5):
    """
    Получение интервалов элементов из предсказаний модели.
    
    Args:
        pred: np.array shape (sequence_length, 2) — вероятности для каждого кадра (фон/элемент)
        threshold: float — порог для класса "элемент"
        
    Returns:
        tuple: (start_frame, end_frame) или str: "элемент не найден"
        
    Raises:
        ValueError: Если входные данные некорректны
    """
    try:
        # Валидация входных данных
        if not isinstance(pred, np.ndarray):
            raise ValueError(f"pred должен быть numpy.ndarray, получен {type(pred)}")
            
        if pred.ndim != 2:
            raise ValueError(f"pred должен быть двумерным массивом, получена размерность {pred.ndim}")
            
        if pred.shape[1] != 2:
            raise ValueError(f"pred должен иметь 2 класса, получено {pred.shape[1]}")
            
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold должен быть в диапазоне [0, 1], получено {threshold}")
        
        # Получаем предсказанные классы для каждого кадра
        pred_classes = np.argmax(pred, axis=-1)  # shape: (sequence_length,)
        
        # Класс 1 — элемент, класс 0 — фон
        element_indices = np.where(pred_classes == 1)[0]
        
        if len(element_indices) == 0:
            return "элемент не найден"
        else:
            # Можно вернуть все интервалы, если элементов несколько, но для простоты — первый и последний
            start_frame = int(element_indices[0])
            end_frame = int(element_indices[-1])
            return start_frame, end_frame
            
    except Exception as e:
        print(f"[ERROR] Ошибка при получении интервалов элементов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

# Пример использования:
# preds = model.predict(batch)  # preds shape: (batch, sequence_length, 2)
# for i, pred in enumerate(preds):
#     try:
#         result = get_element_intervals(pred)
#         print(f"Видео {i}: {result}")
#     except Exception as e:
#         print(f"[ERROR] Ошибка при обработке видео {i}: {str(e)}") 