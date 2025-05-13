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
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import logging
import gc

logger = logging.getLogger(__name__)

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

class ModelTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.network_handler = NetworkErrorHandler()
        self.network_monitor = NetworkMonitor()
        
    def train(self, epochs, batch_size):
        """
        Обучение модели с обработкой сетевых ошибок
        """
        def _train_operation():
            try:
                history = self.model.fit(
                    self.data_loader.data_generator(),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            'best_model.h5',
                            save_best_only=True
                        )
                    ]
                )
                return history
                
            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"Недостаточно памяти GPU: {str(e)}")
                # Попытка освободить память
                tf.keras.backend.clear_session()
                gc.collect()
                raise
                
            except Exception as e:
                logger.error(f"Ошибка при обучении: {str(e)}")
                raise
                
        return self.network_handler.handle_network_operation(_train_operation)

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64):
    """
    Создание модели с обработкой сетевых ошибок
    """
    network_handler = NetworkErrorHandler()
    network_monitor = NetworkMonitor()
    
    def _create_model_operation():
        try:
            # Проверка состояния сети
            network_monitor.check_network_status()
            
            # Входной слой
            inputs = Input(shape=input_shape)
            
            # MobileNetV3Small как backbone
            base_model = MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape[1:],
                include_preprocessing=True
            )
            
            # Проверка доступности GPU
            if not tf.config.list_physical_devices('GPU'):
                logger.warning("GPU не доступен, используется CPU")
                
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
            x = TimeDistributed(Dense(lstm_units // 2, activation='relu'))(x)
            x = Dropout(dropout_rate / 2)(x)
            outputs = TimeDistributed(Dense(num_classes, activation='softmax', dtype='float32'))(x)
            
            return Model(inputs=inputs, outputs=outputs)
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Недостаточно ресурсов GPU: {str(e)}")
            # Попытка освободить память
            tf.keras.backend.clear_session()
            gc.collect()
            raise
            
        except Exception as e:
            logger.error(f"Ошибка при создании модели: {str(e)}")
            raise
            
    return network_handler.handle_network_operation(_create_model_operation)

if __name__ == "__main__":
    try:
        # Инициализация компонентов
        model = create_model(
            input_shape=(16, 224, 224, 3),  # 16 кадров, размер 224x224, 3 канала
            num_classes=2  # Начало и конец элемента
        )
        model.summary()
        
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        # Сохранение состояния модели перед выходом
        if 'model' in locals():
            model.save('emergency_save.h5') 