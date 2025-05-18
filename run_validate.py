import logging
from src.data_proc.advanced_validation import validate_dataset
from src.config import Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Начало валидации датасета...")
    
    # Валидация датасета
    results, stats = validate_dataset()
    
    # Вывод результатов
    logger.info("\n=== Результаты валидации ===")
    logger.info(f"Всего видео: {stats['total_videos']}")
    logger.info(f"Валидных видео: {stats['valid_videos']}")
    logger.info(f"Всего ошибок: {stats['total_errors']}")
    logger.info(f"Всего предупреждений: {stats['total_warnings']}")
    logger.info(f"Всего кадров: {stats['total_frames']}")
    logger.info(f"Кадров действия: {stats['total_action_frames']}")
    logger.info(f"Кадров перехода: {stats['total_transition_frames']}")
    
    # Вывод детальной информации по каждому видео
    logger.info("\n=== Детальная информация по видео ===")
    for video_name, result in results.items():
        logger.info(f"\nВидео: {video_name}")
        if not result.is_valid:
            logger.error("Статус: НЕВАЛИДНО")
            for error in result.errors:
                logger.error(f"Ошибка: {error}")
        else:
            logger.info("Статус: ВАЛИДНО")
        
        for warning in result.warnings:
            logger.warning(f"Предупреждение: {warning}")
        
        if result.stats:
            logger.info("Статистика:")
            logger.info(f"  Всего кадров: {result.stats['total_frames']}")
            logger.info(f"  Кадров действия: {result.stats['action_frames']}")
            logger.info(f"  Кадров перехода: {result.stats['transition_frames']}")
            logger.info(f"  Фоновых кадров: {result.stats['background_frames']}")
            if result.stats['overlaps'] > 0:
                logger.warning(f"  Перекрытий: {result.stats['overlaps']}")
            
            if result.stats['class_distribution']:
                logger.info("  Распределение по типам элементов:")
                for element_type, count in result.stats['class_distribution'].items():
                    logger.info(f"    {element_type}: {count}")

if __name__ == "__main__":
    main() 