import tensorflow as tf
import os
import sys

# Настройка переменных окружения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_DISABLE_JIT'] = '1'

# Проверка версии Python и TensorFlow
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device:", tf.test.gpu_device_name())

# Проверка переменных окружения
print("\nCUDA Environment Variables:")
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

# Проверка наличия библиотек
print("\nChecking CUDA libraries:")
cuda_libs = [
    '/usr/local/cuda/lib64/libcudart.so',
    '/usr/local/cuda/lib64/libcublas.so',
    '/usr/local/cuda/lib64/libcudnn.so',
    '/usr/lib/x86_64-linux-gnu/libcudnn.so'
]

for lib in cuda_libs:
    print(f"{lib}: {'Found' if os.path.exists(lib) else 'Not found'}")

# Тест на GPU
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("\nGPU test successful")
except RuntimeError as e:
    print("\nError during GPU test:", e)