import numpy as np

def get_element_intervals(pred, threshold=0.5):
    """
    pred: np.array shape (sequence_length, 2) — вероятности для каждого кадра (фон/элемент)
    threshold: float — порог для класса "элемент"
    Возвращает: (start_frame, end_frame) или "элемент не найден"
    """
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

# Пример использования:
# preds = model.predict(batch)  # preds shape: (batch, sequence_length, 2)
# for i, pred in enumerate(preds):
#     result = get_element_intervals(pred)
#     print(f"Видео {i}: {result}") 