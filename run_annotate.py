from src.data_proc.annotate_video import process_all_videos, process_validation_videos

if __name__ == "__main__":
    # Обрабатываем тренировочные видео
    process_all_videos()
    
    # Обрабатываем валидационные видео
    process_validation_videos() 