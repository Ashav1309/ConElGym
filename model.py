import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout
)
from tensorflow.keras.applications import MobileNetV2

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = Conv2D(1, kernel_size, padding='same')
        self.bn = BatchNormalization()
        
    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=3)
        x = self.conv(x)
        x = self.bn(x)
        return tf.sigmoid(x)

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        
    def call(self, query, values):
        score = tf.nn.tanh(self.W1(query) + self.W2(values))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * values
        return context_vector, attention_weights

def create_model(input_shape, num_classes):
    # Входной слой
    inputs = Input(shape=input_shape)
    
    # MobileNetV2 как backbone
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # Извлечение признаков
    x = base_model(inputs)
    
    # Пространственное внимание
    spatial_attention = SpatialAttention()(x)
    x = Multiply()([x, spatial_attention])
    
    # Подготовка для LSTM
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, -1))(x)
    
    # BiLSTM слои
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Временное внимание
    temporal_attention = TemporalAttention(128)
    context_vector, attention_weights = temporal_attention(x, x)
    
    # Выходные слои
    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    # Пример использования
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=2  # Начало и конец элемента
    )
    model.summary() 