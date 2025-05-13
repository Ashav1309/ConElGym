import tensorflow as tf
import time
import gc
import logging

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
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                return operation(*args, **kwargs)
            except (tf.errors.UnavailableError, 
                   tf.errors.DeadlineExceededError,
                   tf.errors.ResourceExhaustedError) as e:
                last_error = e
                retry_count += 1
                logger.warning(f"Сетевая ошибка: {str(e)}. Попытка {retry_count}/{self.max_retries}")
                time.sleep(self.retry_delay * retry_count)  # Увеличиваем задержку с каждой попыткой
                
        raise last_error

class NetworkMonitor:
    def __init__(self):
        self.last_check = time.time()
        self.check_interval = 60  # Проверка каждую минуту
        
    def check_network_status(self):
        """
        Проверка состояния сети
        """
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            try:
                # Проверка доступности GPU
                gpu_devices = tf.config.list_physical_devices('GPU')
                if not gpu_devices:
                    logger.warning("GPU недоступен")
                    
                # Проверка памяти GPU
                if gpu_devices:
                    gpu = tf.config.experimental.get_visible_devices('GPU')[0]
                    memory_info = tf.config.experimental.get_memory_info(gpu)
                    logger.info(f"Использование памяти GPU: {memory_info['current'] / 1024**2:.2f}MB")
                    
                self.last_check = current_time
                
            except Exception as e:
                logger.error(f"Ошибка при проверке состояния сети: {str(e)}") 