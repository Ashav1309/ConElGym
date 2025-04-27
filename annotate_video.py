import cv2
import os
from annotation import VideoAnnotation
from config import Config
from datetime import datetime

class VideoAnnotator:
    def __init__(self, video_path: str, is_validation: bool = False):
        """
        Инициализация класса для аннотирования видео.
        
        Args:
            video_path (str): Путь к видеофайлу для аннотирования
            is_validation (bool): Флаг, указывающий что это валидационные данные
        """
        self.video_path = video_path
        self.is_validation = is_validation
        self.cap = cv2.VideoCapture(video_path)
        self.annotation = VideoAnnotation(video_path)
        self.current_frame = 0
        self.paused = False
        self.start_frame = None
        
    def process_frame(self, frame):
        """
        Обработка и отображение кадра с дополнительной информацией.
        
        Args:
            frame: Кадр видео для обработки
            
        Returns:
            Обработанный кадр с наложенной информацией
        """
        # Отображение номера текущего кадра
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Отображение инструкций по управлению
        cv2.putText(frame, "Space: pause/resume", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "S: mark start", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "E: mark end", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "R: restart", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Q: quit", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """
        Основной цикл аннотирования видео.
        """
        # Получаем FPS видео
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Устанавливаем задержку между кадрами (в миллисекундах)
        frame_delay = int(1000 / (fps * 0.5))  # Замедляем в 2 раза
        
        while True:
            # Если видео не на паузе, читаем следующий кадр
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:  # Если видео закончилось
                    break
                    
                frame = self.process_frame(frame)  # Обрабатываем кадр
                cv2.imshow('Video Annotation', frame)  # Отображаем кадр
                self.current_frame += 1  # Увеличиваем счетчик кадров
                
            # Обработка нажатий клавиш с задержкой
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord(' '):  # Пауза/продолжение воспроизведения
                self.paused = not self.paused
            elif key == ord('s'):  # Отметить начало элемента
                self.start_frame = self.current_frame
                print(f"Start frame marked: {self.start_frame}")
            elif key == ord('e'):  # Отметить конец элемента
                if self.start_frame is not None:
                    # Добавляем элемент в аннотации
                    self.annotation.add_element(
                        self.start_frame,
                        self.current_frame,
                        "element"  # Здесь можно указать тип элемента
                    )
                    print(f"Element marked: {self.start_frame} - {self.current_frame}")
                    self.start_frame = None  # Сбрасываем начало элемента
            elif key == ord('r'):  # Перезапуск видео
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Устанавливаем позицию на начало
                self.current_frame = 0  # Сбрасываем счетчик кадров
                print("Видео перезапущено")
            elif key == ord('q'):  # Выход из программы
                break
                
        # Освобождаем ресурсы
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Сохранение разметки в JSON файл
        video_name = os.path.basename(self.video_path)
        base_path = Config.VALID_DATA_PATH if self.is_validation else Config.TRAIN_DATA_PATH
        
        annotation_path = os.path.join(
            base_path,
            'annotations',
            f"{os.path.splitext(video_name)[0]}.json"
        )
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        self.annotation.save(annotation_path)
        
        # Запись информации о размеченном видео в лог-файл
        self._log_annotation(video_name)

    def _log_annotation(self, video_name: str):
        """
        Запись информации о размеченном видео в лог-файл.
        """
        base_path = Config.VALID_DATA_PATH if self.is_validation else Config.TRAIN_DATA_PATH
        log_file = os.path.join(base_path, 'annotation_log.txt')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{timestamp} - {video_name}\n")
            f.write(f"Тип данных: {'валидационные' if self.is_validation else 'тренировочные'}\n")
            f.write(f"Количество размеченных элементов: {len(self.annotation.annotations)}\n")
            for i, ann in enumerate(self.annotation.annotations, 1):
                f.write(f"Элемент {i}: кадры {ann['start_frame']} - {ann['end_frame']}\n")
            f.write("-" * 50 + "\n")

def get_annotated_videos(is_validation: bool = False):
    """
    Получение списка уже размеченных видео из лог-файла.
    
    Args:
        is_validation (bool): Флаг для проверки валидационных данных
        
    Returns:
        set: Множество имен размеченных видеофайлов
    """
    base_path = Config.VALID_DATA_PATH if is_validation else Config.TRAIN_DATA_PATH
    log_file = os.path.join(base_path, 'annotation_log.txt')
    annotated_videos = set()
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and '-' in line:
                    video_name = line.split('-')[1].strip()
                    annotated_videos.add(video_name)
    
    return annotated_videos

def process_all_videos():
    """
    Обработка всех видео из папки data/train.
    """
    train_dir = Config.TRAIN_DATA_PATH  # Убираем лишний 'train' в пути
    
    # Проверяем существование директории
    if not os.path.exists(train_dir):
        print(f"Директория {train_dir} не существует")
        return
    
    # Получаем список всех видеофайлов и уже размеченных видео
    video_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.mp4')]
    annotated_videos = get_annotated_videos(is_validation=False)
    
    # Фильтруем видео, оставляя только неразмеченные
    video_files = [f for f in video_files if f not in annotated_videos]
    
    if not video_files:
        print("Нет новых видео для обработки")
        return
    
    print(f"Найдено {len(video_files)} новых видеофайлов для обработки")
    
    # Обрабатываем каждое видео
    for video_file in video_files:
        video_path = os.path.join(train_dir, video_file)
        print(f"\nОбработка видео: {video_file}")
        
        try:
            annotator = VideoAnnotator(video_path, is_validation=False)
            annotator.run()
            print(f"Видео {video_file} успешно обработано")
        except Exception as e:
            print(f"Ошибка при обработке видео {video_file}: {str(e)}")

def process_validation_videos():
    """
    Обработка всех видео из папки data/valid.
    """
    valid_dir = Config.VALID_DATA_PATH
    
    # Проверяем существование директории
    if not os.path.exists(valid_dir):
        print(f"Директория {valid_dir} не существует")
        return
    
    # Получаем список всех видеофайлов и уже размеченных видео
    video_files = [f for f in os.listdir(valid_dir) if f.lower().endswith('.mp4')]
    annotated_videos = get_annotated_videos(is_validation=True)
    
    # Фильтруем видео, оставляя только неразмеченные
    video_files = [f for f in video_files if f not in annotated_videos]
    
    if not video_files:
        print("Нет новых валидационных видео для обработки")
        return
    
    print(f"Найдено {len(video_files)} новых валидационных видеофайлов для обработки")
    
    # Обрабатываем каждое видео
    for video_file in video_files:
        video_path = os.path.join(valid_dir, video_file)
        print(f"\nОбработка валидационного видео: {video_file}")
        
        try:
            annotator = VideoAnnotator(video_path, is_validation=True)
            annotator.run()
            print(f"Валидационное видео {video_file} успешно обработано")
        except Exception as e:
            print(f"Ошибка при обработке валидационного видео {video_file}: {str(e)}")

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs(Config.TRAIN_DATA_PATH, exist_ok=True)
    os.makedirs(Config.VALID_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.join(Config.TRAIN_DATA_PATH, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(Config.VALID_DATA_PATH, 'annotations'), exist_ok=True)
    
    # Обрабатываем тренировочные видео
    process_all_videos()
    
    # Обрабатываем валидационные видео
    process_validation_videos() 