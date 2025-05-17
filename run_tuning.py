import os
import sys
from src.models.hyperparameter_tuning import tune_hyperparameters

# Настройка переменных окружения для GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уменьшаем уровень логирования
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Используем первую GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Импортируем TensorFlow после настройки переменных окружения
import tensorflow as tf

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

def objective(trial):
    try:
        print(f"\n[DEBUG] ===== ============================ ======")
        print(f"\n[DEBUG] ===== Начало триала #{trial.number} =====")
        print(f"\n[DEBUG] ===== ============================ ======")
        # Получаем гиперпараметры из trial
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 32, 256)
        model_type = Config.MODEL_TYPE
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'bigru'])
        temporal_block_type = trial.suggest_categorical('temporal_block_type', ['rnn', 'hybrid', '3d_attention', 'transformer'])
        clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
        
        print(f"[DEBUG] Параметры триала #{trial.number}:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        print(f"  - lstm_units: {lstm_units}")
        print(f"  - rnn_type: {rnn_type}")
        print(f"  - temporal_block_type: {temporal_block_type}")
        print(f"  - clipnorm: {clipnorm}")
        # ... existing code ...
    except Exception as e:
        print(f"\nError during hyperparameter tuning: {str(e)}")
        sys.exit(1)

def main():
    try:
        result = tune_hyperparameters()
        if result is not None:
            print("\nBest parameters:", result['best_params'])
            print("Best validation accuracy:", result['best_value'])
            
            # Сохраняем лучшие параметры в файл
            with open('best_params.txt', 'w') as f:
                f.write("Best parameters:\n")
                for param, value in result['best_params'].items():
                    f.write(f"{param}: {value}\n")
                f.write(f"\nBest validation accuracy: {result['best_value']:.4f}\n")
        else:
            print("\nFailed to find best parameters. Check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError during hyperparameter tuning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 