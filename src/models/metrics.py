import tensorflow as tf

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