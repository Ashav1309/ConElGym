import tensorflow as tf
from src.models.losses import F1ScoreAdapter

def f1_score_element(y_true, y_pred):
    """
    Вычисление F1-score для элемента с учетом временной размерности и two-hot encoded меток
    """
    # Получаем предсказания для класса действия
    y_true_bin = y_true[:, :, 1]  # Класс действия
    y_pred_bin = y_pred[:, :, 1]  # Класс действия
    
    true_positives = tf.reduce_sum(tf.cast((y_true_bin == 1) & (y_pred_bin == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred_bin == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true_bin == 1, tf.float32))
    
    # Добавляем epsilon для предотвращения деления на ноль
    epsilon = tf.keras.backend.epsilon()
    
    # Вычисляем precision и recall
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)
    
    # Вычисляем F1-score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

def get_training_metrics():
    """
    Получение метрик для обучения модели
    """
    return [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5),
        tf.keras.metrics.AUC(name='auc'),
        F1ScoreAdapter(name='f1_score_element', threshold=0.5)
    ]

def get_tuning_metrics():
    """
    Получение метрик для подбора гиперпараметров
    """
    class SequenceMetrics(tf.keras.metrics.Metric):
        def __init__(self, name='sequence_metrics', **kwargs):
            super().__init__(name=name, **kwargs)
            self.precision = tf.keras.metrics.Precision(thresholds=0.5)
            self.recall = tf.keras.metrics.Recall(thresholds=0.5)
            self.f1 = tf.keras.metrics.F1Score(threshold=0.5)
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            # Преобразуем входные данные в 2D
            batch_size = tf.shape(y_true)[0]
            y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            
            if sample_weight is not None:
                sample_weight = tf.reshape(sample_weight, [-1])
            
            # Обновляем метрики
            self.precision.update_state(y_true, y_pred, sample_weight)
            self.recall.update_state(y_true, y_pred, sample_weight)
            self.f1.update_state(y_true, y_pred, sample_weight)
            
        def result(self):
            return {
                'precision': self.precision.result(),
                'recall': self.recall.result(),
                'f1_score': self.f1.result()
            }
            
        def reset_state(self):
            self.precision.reset_state()
            self.recall.reset_state()
            self.f1.reset_state()
    
    return [
        'accuracy',
        SequenceMetrics(name='sequence_metrics')
    ] 