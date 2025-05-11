import tensorflow as tf
import os
from src.models.hyperparameter_tuning import tune_hyperparameters

# Включаем подробный вывод
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Настройка GPU
print("TensorFlow version:", tf.__version__)
print("CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device:", tf.test.gpu_device_name())

# Проверяем переменные окружения CUDA
print("\nCUDA Environment Variables:")
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Разрешаем динамический рост памяти
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\nMemory growth enabled for GPUs")
        
        # Устанавливаем видимые устройства
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Set visible GPU device")
        
        # Включаем mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("\nNo GPU devices found")

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 