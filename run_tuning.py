import tensorflow as tf
from src.models.hyperparameter_tuning import tune_hyperparameters

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Включение mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 