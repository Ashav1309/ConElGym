import tensorflow as tf
import time
import gc
import logging
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkErrorHandler:
    def __init__(self, max_retries=3, retry_delay=1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def handle_network_operation(self, operation, *args, **kwargs):
        """
        Обработка сетевых операций с повторными попытками
        
        Args:
            operation: Функция для выполнения
            *args: Аргументы функции
            **kwargs: Именованные аргументы функции
            
        Returns:
            Результат выполнения операции
            
        Raises:
            Exception: Если все попытки выполнения операции завершились неудачей
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                return operation(*args, **kwargs)
            except (tf.errors.UnavailableError, 
                   tf.errors.DeadlineExceededError,
                   tf.errors.ResourceExhaustedError,
                   tf.errors.FailedPreconditionError,
                   tf.errors.AbortedError,
                   tf.errors.InternalError,
                   tf.errors.UnknownError) as e:
                last_error = e
                retry_count += 1
                logger.warning(f"Сетевая ошибка: {str(e)}. Попытка {retry_count}/{self.max_retries}")
                
                # Очистка памяти перед повторной попыткой
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Увеличиваем задержку с каждой попыткой
                time.sleep(self.retry_delay * retry_count)
                
        logger.error(f"Все попытки выполнения операции завершились неудачей. Последняя ошибка: {str(last_error)}")
        raise last_error

class NetworkMonitor:
    def __init__(self):
        self.last_check = time.time()
        self.check_interval = 60  # Проверка каждую минуту
        self.memory_threshold = 0.9  # Порог использования памяти (90%)
        
    def check_network_status(self):
        """
        Проверка состояния сети и ресурсов
        
        Returns:
            dict: Словарь с информацией о состоянии системы
        """
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            try:
                status = {
                    'timestamp': current_time,
                    'gpu_available': False,
                    'gpu_memory_used': 0,
                    'gpu_memory_total': 0,
                    'cpu_memory_used': 0,
                    'cpu_memory_total': 0,
                    'disk_usage': 0
                }
                
                # Проверка доступности GPU
                gpu_devices = tf.config.list_physical_devices('GPU')
                status['gpu_available'] = bool(gpu_devices)
                
                if not gpu_devices:
                    logger.warning("GPU недоступен")
                else:
                    # Проверка памяти GPU
                    gpu = tf.config.experimental.get_visible_devices('GPU')[0]
                    memory_info = tf.config.experimental.get_memory_info(gpu)
                    status['gpu_memory_used'] = memory_info['current'] / 1024**2  # MB
                    status['gpu_memory_total'] = memory_info['peak'] / 1024**2  # MB
                    
                    # Проверка использования памяти GPU
                    gpu_memory_usage = status['gpu_memory_used'] / status['gpu_memory_total']
                    if gpu_memory_usage > self.memory_threshold:
                        logger.warning(f"Высокое использование памяти GPU: {gpu_memory_usage:.2%}")
                
                # Проверка памяти CPU
                cpu_memory = psutil.virtual_memory()
                status['cpu_memory_used'] = cpu_memory.used / 1024**2  # MB
                status['cpu_memory_total'] = cpu_memory.total / 1024**2  # MB
                
                # Проверка использования памяти CPU
                cpu_memory_usage = cpu_memory.percent / 100
                if cpu_memory_usage > self.memory_threshold:
                    logger.warning(f"Высокое использование памяти CPU: {cpu_memory_usage:.2%}")
                
                # Проверка использования диска
                disk_usage = psutil.disk_usage('/')
                status['disk_usage'] = disk_usage.percent / 100
                
                if status['disk_usage'] > self.memory_threshold:
                    logger.warning(f"Высокое использование диска: {status['disk_usage']:.2%}")
                
                self.last_check = current_time
                return status
                
            except Exception as e:
                logger.error(f"Ошибка при проверке состояния сети: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                return None 