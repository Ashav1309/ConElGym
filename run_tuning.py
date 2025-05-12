import tensorflow as tf
import os
from src.models.hyperparameter_tuning import tune_hyperparameters

# Настройка переменных окружения для GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_DISABLE_JIT'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# Отключаем JIT компиляцию
tf.config.optimizer.set_jit(False)

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
        # Сбрасываем статистику памяти GPU
        tf.config.experimental.reset_memory_stats('GPU:0')
        
        # Включаем mixed precision
        from tensorflow.keras.mixed_precision import Policy
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set:", policy.name)
        
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
    try:
        study = tune_hyperparameters()
        if study.best_trial.value is not None:
            print("\nЛучшие параметры:")
            print(f"Learning rate: {study.best_trial.params['learning_rate']:.6f}")
            print(f"Dropout rate: {study.best_trial.params['dropout_rate']:.4f}")
            print(f"LSTM units: {study.best_trial.params['lstm_units']}")
            print(f"Validation accuracy: {study.best_trial.value:.4f}")
        else:
            print("\nНе удалось найти лучшие параметры. Все испытания завершились с ошибкой.")
    except Exception as e:
        print(f"Ошибка при подборе гиперпараметров: {e}")
        raise 