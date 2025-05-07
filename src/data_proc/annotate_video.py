import cv2
import os
from src.data_proc.annotation import VideoAnnotation
from src.config import Config
import json
from datetime import datetime
import numpy as np

class VideoAnnotator:
    def __init__(self, video_path: str, is_validation: bool = False, video_list: list = None):
        """
        Инициализация класса для аннотирования видео.
        
        Args:
            video_path (str): Путь к видеофайлу для аннотирования
            is_validation (bool): Флаг, указывающий что это валидационные данные
            video_list (list): Список всех видео для навигации
        """
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.is_validation = is_validation
        self.cap = cv2.VideoCapture(video_path)
        self.annotation = VideoAnnotation(video_path)
        self.current_frame = 0
        self.paused = False
        self.start_frame = None
        self.video_list = video_list or []
        self.current_video_index = self.video_list.index(video_path) if video_path in self.video_list else -1
        self.annotated_videos = get_annotated_videos(is_validation)
        self.previous_annotations = []  # Список для хранения предыдущих аннотаций
        
    def process_frame(self, frame):
        """
        Обработка и отображение кадра с дополнительной информацией.
        
        Args:
            frame: Кадр видео для обработки
            
        Returns:
            Обработанный кадр с наложенной информацией
        """
        # Отображение названия видео
        cv2.putText(frame, f"Video: {self.video_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Отображение номера текущего кадра
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Отображение текущих аннотаций
        y_offset = 90
        cv2.putText(frame, "Current annotations:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for i, ann in enumerate(self.annotation.annotations, 1):
            # Проверяем, находится ли текущий кадр в пределах аннотации
            if self.start_frame is not None and self.current_frame >= self.start_frame:
                color = (0, 255, 0)  # Зеленый для текущей разметки
            else:
                color = (255, 255, 255)  # Белый для остальных
            
            # Отображаем информацию об аннотации
            text = f"{i}. Frames: {ann['start_frame']} - {ann['end_frame']}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Отображение инструкций по управлению
        y_offset = max(y_offset, 300)  # Минимальный отступ для инструкций
        cv2.putText(frame, "Space: pause/resume", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "S: mark start", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "E: mark end", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "R: restart", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "B: previous video", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "N: next video", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, "Q: quit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def save_current_annotation(self):
        """Сохраняет текущую аннотацию в список предыдущих"""
        if self.annotation.annotations:
            self.previous_annotations.append(self.annotation.annotations.copy())
            print(f"Сохранена аннотация #{len(self.previous_annotations)}")

    def load_previous_annotation(self):
        """Загружает предыдущую аннотацию"""
        if self.previous_annotations:
            self.annotation.annotations = self.previous_annotations.pop()
            print(f"Загружена предыдущая аннотация")
        else:
            print("Нет сохраненных аннотаций")

    def get_next_unannotated_video(self, direction: int = 1):
        """
        Находит следующее неаннотированное видео в указанном направлении.
        
        Args:
            direction (int): 1 для следующего видео, -1 для предыдущего
            
        Returns:
            str or None: Путь к следующему неаннотированному видео или None, если такого нет
        """
        if not self.video_list:
            return None
            
        current_idx = self.current_video_index
        while True:
            current_idx += direction
            if current_idx < 0 or current_idx >= len(self.video_list):
                return None
                
            next_video = self.video_list[current_idx]
            video_name = os.path.basename(next_video).lower()
            if video_name not in self.annotated_videos:
                return next_video

    def run(self):
        """
        Основной цикл аннотирования видео.
        """
        # Проверяем, не аннотировано ли текущее видео
        if os.path.basename(self.video_path).lower() in self.annotated_videos:
            print(f"Видео {self.video_name} уже аннотировано. Пропускаем.")
            return self.get_next_unannotated_video(1)
            
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
            elif key == ord('b'):  # Предыдущее видео
                next_video = self.get_next_unannotated_video(-1)
                if next_video:
                    self._save_annotation()  # Сохраняем текущую аннотацию
                    return next_video
                else:
                    print("Нет предыдущих неаннотированных видео")
            elif key == ord('n'):  # Следующее видео
                next_video = self.get_next_unannotated_video(1)
                if next_video:
                    self._save_annotation()  # Сохраняем текущую аннотацию
                    return next_video
                else:
                    print("Нет следующих неаннотированных видео")
            elif key == ord('q'):  # Выход из программы
                self._save_annotation()  # Сохраняем текущую аннотацию
                return None
                
        # Освобождаем ресурсы
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Сохраняем аннотацию перед переходом к следующему видео
        self._save_annotation()
        
        # Переходим к следующему неаннотированному видео
        return self.get_next_unannotated_video(1)

    def _save_annotation(self):
        """Сохраняет текущую аннотацию в файл"""
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
    Получение списка уже размеченных видео из папки annotations.
    
    Args:
        is_validation (bool): Флаг для проверки валидационных данных
        
    Returns:
        set: Множество имен размеченных видеофайлов
    """
    base_path = Config.VALID_DATA_PATH if is_validation else Config.TRAIN_DATA_PATH
    annotations_dir = os.path.join(base_path, 'annotations')
    annotated_videos = set()
    
    if os.path.exists(annotations_dir):
        # Получаем список всех JSON файлов в папке annotations
        json_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.json')]
        
        # Преобразуем имена JSON файлов в имена видеофайлов
        for json_file in json_files:
            # Заменяем расширение .json на .mp4
            video_name = json_file.replace('.json', '.mp4')
            annotated_videos.add(video_name.lower())
    
    return annotated_videos

def process_all_videos():
    """
    Обработка всех видео из папки data/train.
    """
    train_dir = Config.TRAIN_DATA_PATH
    
    # Проверяем существование директории
    if not os.path.exists(train_dir):
        print(f"Директория {train_dir} не существует")
        return
    
    # Получаем список всех видеофайлов
    all_video_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.mp4')]
    
    # Получаем список уже размеченных видео
    annotated_videos = get_annotated_videos(is_validation=False)
    print(f"Найдено {len(annotated_videos)} уже размеченных видео")
    print("Аннотированные видео:")
    for video in annotated_videos:
        print(f"- {video}")
    
    # Фильтруем видео, оставляя только неразмеченные
    video_files = [f for f in all_video_files if f.lower() not in annotated_videos]
    video_paths = [os.path.join(train_dir, f) for f in video_files]
    
    if not video_paths:
        print("Нет новых видео для обработки")
        return
    
    print(f"\nНайдено {len(video_paths)} новых видеофайлов для обработки")
    print("Список новых видео:")
    for video in video_files:
        print(f"- {video}")
    
    # Обрабатываем видео
    current_video = video_paths[0]
    while current_video is not None:
        print(f"\nОбработка видео: {os.path.basename(current_video)}")
        try:
            annotator = VideoAnnotator(current_video, is_validation=False, video_list=video_paths)
            current_video = annotator.run()
        except Exception as e:
            print(f"Ошибка при обработке видео {os.path.basename(current_video)}: {str(e)}")
            break

def process_validation_videos():
    """
    Обработка всех видео из папки data/valid.
    """
    valid_dir = Config.VALID_DATA_PATH
    
    # Проверяем существование директории
    if not os.path.exists(valid_dir):
        print(f"Директория {valid_dir} не существует")
        return
    
    # Получаем список всех видеофайлов
    all_video_files = [f for f in os.listdir(valid_dir) if f.lower().endswith('.mp4')]
    
    # Получаем список уже размеченных видео
    annotated_videos = get_annotated_videos(is_validation=True)
    print(f"Найдено {len(annotated_videos)} уже размеченных видео")
    print("Аннотированные видео:")
    for video in annotated_videos:
        print(f"- {video}")
    
    # Фильтруем видео, оставляя только неразмеченные
    video_files = [f for f in all_video_files if f.lower() not in annotated_videos]
    video_paths = [os.path.join(valid_dir, f) for f in video_files]
    
    if not video_paths:
        print("Нет новых валидационных видео для обработки")
        return
    
    print(f"\nНайдено {len(video_paths)} новых валидационных видеофайлов для обработки")
    print("Список новых видео:")
    for video in video_files:
        print(f"- {video}")
    
    # Обрабатываем видео
    current_video = video_paths[0]
    while current_video is not None:
        print(f"\nОбработка валидационного видео: {os.path.basename(current_video)}")
        try:
            annotator = VideoAnnotator(current_video, is_validation=True, video_list=video_paths)
            current_video = annotator.run()
        except Exception as e:
            print(f"Ошибка при обработке валидационного видео {os.path.basename(current_video)}: {str(e)}")
            break

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