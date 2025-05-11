import tensorflow as tf
import os
from src.models.hyperparameter_tuning import tune_hyperparameters

# Настройка переменных окружения для GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
        # Включаем mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
        
        # Проверяем, что GPU действительно используется
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("GPU test successful")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("\nNo GPU devices found")

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 