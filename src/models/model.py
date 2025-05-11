import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout, TimeDistributed,
    GlobalAveragePooling1D
)
from tensorflow.keras.applications import MobileNetV3Small

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.conv = None
        self.bn = None
        
    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, padding='same')
        self.bn = BatchNormalization()
        super(SpatialAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv(x)
        x = self.bn(x)
        return tf.sigmoid(x)

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.units = units
        self.W1 = None
        self.W2 = None
        self.V = None
        
    def build(self, input_shape):
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)
        super(TemporalAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x):
        # Используем один и тот же вход для query и values
        query = x
        values = x
        
        # Вычисляем attention scores
        score = tf.nn.tanh(self.W1(query) + self.W2(values))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Применяем веса к значениям
        context_vector = attention_weights * values
        
        # Возвращаем только context_vector
        return context_vector

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64):
    """
    Создание модели для анализа видео.
    
    Args:
        input_shape (tuple): Размерность входных данных (sequence_length, height, width, channels)
        num_classes (int): Количество классов для классификации
        dropout_rate (float): Коэффициент dropout
        lstm_units (int): Количество нейронов в LSTM слое
    """
    # Входной слой
    inputs = Input(shape=input_shape)
    
    # MobileNetV3Small как backbone
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape[1:],  # Используем размерность одного кадра
        include_preprocessing=True
    )
    base_model.trainable = False
    
    # Применяем MobileNetV3 к каждому кадру в последовательности
    x = TimeDistributed(base_model)(inputs)
    
    # Пространственное внимание для каждого кадра
    spatial_attention = TimeDistributed(SpatialAttention())(x)
    x = Multiply()([x, spatial_attention])
    
    # Подготовка для LSTM
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # BiLSTM слои
    x = Bidirectional(LSTM(lstm_units * 2, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    
    # Временное внимание
    x = TemporalAttention(lstm_units)(x)
    
    # Глобальное усреднение по временной оси
    x = GlobalAveragePooling1D()(x)
    
    # Выходные слои
    x = Dense(lstm_units // 2, activation='relu')(x)
    x = Dropout(dropout_rate / 2)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Изменено на softmax для многоклассовой классификации
    
    return Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    # Пример использования
    model = create_model(
        input_shape=(16, 224, 224, 3),  # 16 кадров, размер 224x224, 3 канала
        num_classes=2  # Начало и конец элемента
    )
    model.summary() 