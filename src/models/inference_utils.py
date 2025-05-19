import numpy as np

def get_element_intervals(pred, threshold=0.5, fps=25):
    """
    Получение интервалов элементов из предсказаний модели в секундах.
    
    Args:
        pred: np.array shape (sequence_length, 2) — вероятности для каждого кадра (фон/элемент)
        threshold: float — порог для класса "элемент"
        fps: int или float — частота кадров (по умолчанию 25)
        
    Returns:
        list: список (start_sec, end_sec) для каждого действия
        или str: "элемент не найден"
    """
    try:
        if not isinstance(pred, np.ndarray):
            raise ValueError(f"pred должен быть numpy.ndarray, получен {type(pred)}")
        if pred.ndim != 2:
            raise ValueError(f"pred должен быть двумерным массивом, получена размерность {pred.ndim}")
        if pred.shape[1] != 2:
            raise ValueError(f"pred должен иметь 2 класса, получено {pred.shape[1]}")
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold должен быть в диапазоне [0, 1], получено {threshold}")
        
        pred_classes = np.argmax(pred, axis=-1)
        element_mask = (pred_classes == 1)
        
        # Находим интервалы подряд идущих True
        intervals = []
        in_interval = False
        start = None
        for i, is_action in enumerate(element_mask):
            if is_action and not in_interval:
                in_interval = True
                start = i
            elif not is_action and in_interval:
                in_interval = False
                end = i - 1
                # Переводим в секунды
                start_sec = start / fps
                end_sec = end / fps
                intervals.append((start_sec, end_sec))
        # Если последовательность закончилась на действии
        if in_interval:
            end = len(element_mask) - 1
            start_sec = start / fps
            end_sec = end / fps
            intervals.append((start_sec, end_sec))
        
        if not intervals:
            return "элемент не найден"
        else:
            return intervals
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