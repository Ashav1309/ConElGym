import os
import tensorflow as tf
from src.config import Config

def setup_gpu():
    """
    Настройка GPU для TensorFlow
    Включает:
    - Настройку переменных окружения
    - Включение динамического роста памяти
    - Настройку mixed precision
    - Отключение JIT компиляции
    - Настройку CPU/GPU в зависимости от конфигурации
    """
    if Config.DEVICE_CONFIG['use_gpu']:
        # Настройка переменных окружения для GPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Фильтрация логов TensorFlow
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Используем первую GPU
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['TF_DISABLE_JIT'] = '1'
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

        # Включаем eager execution
        tf.config.run_functions_eagerly(True)

        # Настройка GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Включаем динамический рост памяти для всех GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, Config.DEVICE_CONFIG['allow_gpu_memory_growth'])
                print(f"[DEBUG] Включён динамический рост памяти для {len(gpus)} GPU")
            except RuntimeError as e:
                print(f"[ERROR] Ошибка при настройке GPU: {e}")
                return False

        # Включение mixed precision если нужно
        if Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
            from tensorflow.keras.mixed_precision import Policy
            policy = Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision policy set:", policy.name)

        # Отключаем JIT компиляцию
        tf.config.optimizer.set_jit(False)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("GPU Device: ", tf.test.gpu_device_name())
        print("GPU optimization enabled")
        return True
    else:
        # Настройка CPU
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(Config.DEVICE_CONFIG['cpu_threads'])
        tf.config.threading.set_inter_op_parallelism_threads(Config.DEVICE_CONFIG['cpu_threads'])
        print("CPU optimization enabled")
        return True 