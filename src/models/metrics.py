import tensorflow as tf
from src.models.losses import F1ScoreAdapter
from src.models.callbacks import ScalarF1Score

def get_training_metrics():
    """
    Получение метрик для обучения модели.
    Все метрики рассчитываются для класса действия (class_id=1).
    """
    return [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
        F1ScoreAdapter(name='f1_action', class_id=1, threshold=0.5)
    ]

def get_tuning_metrics():
    """
    Получение метрик для подбора гиперпараметров.
    Использует те же метрики, что и для обучения.
    """
    return get_training_metrics()

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Расчет всех метрик для заданных предсказаний.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказания модели
        threshold: Порог классификации
        
    Returns:
        dict: Словарь с метриками
    """
    # Бинаризация предсказаний
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    # Получаем метрики для класса действия
    precision = tf.keras.metrics.Precision(thresholds=threshold)
    recall = tf.keras.metrics.Recall(thresholds=threshold)
    f1 = F1ScoreAdapter(threshold=threshold)
    
    # Обновляем состояние метрик
    precision.update_state(y_true[..., 1], y_pred[..., 1])
    recall.update_state(y_true[..., 1], y_pred[..., 1])
    f1.update_state(y_true, y_pred)
    
    return {
        'accuracy': tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)),
        'precision_action': precision.result(),
        'recall_action': recall.result(),
        'f1_action': f1.result()
    } 