import json
import os
from typing import List, Dict, Tuple
import cv2

class VideoAnnotation:
    def __init__(self, video_path: str):
        """
        Инициализация класса для работы с аннотациями видео.
        
        Args:
            video_path (str): Путь к видеофайлу
        """
        self.video_path = video_path  # Путь к видеофайлу
        self.annotations = []  # Список аннотаций элементов
        self.video_name = os.path.basename(video_path)  # Имя видеофайла
        
    def add_element(self, start_frame: int, end_frame: int, 
                   element_type: str, confidence: float = 1.0):
        """
        Добавление разметки элемента в список аннотаций.
        
        Args:
            start_frame (int): Номер кадра начала элемента
            end_frame (int): Номер кадра окончания элемента
            element_type (str): Тип элемента
            confidence (float): Уверенность в разметке (по умолчанию 1.0)
        """
        annotation = {
            'start_frame': start_frame,  # Кадр начала
            'end_frame': end_frame,      # Кадр окончания
            'element_type': element_type, # Тип элемента
            'confidence': confidence      # Уверенность в разметке
        }
        self.annotations.append(annotation)
        
    def save(self, output_path: str):
        """
        Сохранение разметки в JSON файл.
        
        Args:
            output_path (str): Путь для сохранения JSON файла
        """
        data = {
            'video_name': self.video_name,  # Имя видеофайла
            'annotations': self.annotations  # Список аннотаций
        }
        
        # Сохранение в JSON с форматированием
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    @classmethod
    def load(cls, annotation_path: str, video_path: str):
        """
        Загрузка разметки из JSON файла.
        
        Args:
            annotation_path (str): Путь к JSON файлу с разметкой
            video_path (str): Путь к видеофайлу
            
        Returns:
            VideoAnnotation: Объект с загруженной разметкой
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Создание нового объекта и загрузка данных
        annotation = cls(video_path)
        annotation.annotations = data['annotations']
        return annotation
    
    def get_frame_labels(self, frame_number: int) -> Tuple[bool, bool]:
        """
        Получение меток для конкретного кадра.
        
        Args:
            frame_number (int): Номер кадра
            
        Returns:
            Tuple[bool, bool]: (is_start, is_end) - метки начала и конца элемента
        """
        is_start = False  # Флаг начала элемента
        is_end = False    # Флаг окончания элемента
        
        # Проверка всех аннотаций
        for ann in self.annotations:
            if frame_number == ann['start_frame']:
                is_start = True
            if frame_number == ann['end_frame']:
                is_end = True
                
        return is_start, is_end
    
    def visualize(self, output_path: str):
        """
        Визуализация разметки на видео.
        Создает новое видео с наложенными метками начала и окончания элементов.
        
        Args:
            output_path (str): Путь для сохранения видео с разметкой
        """
        # Открытие видеофайла
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Получение FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
        
        # Создание объекта для записи видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Обработка каждого кадра
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Получение меток для текущего кадра
            is_start, is_end = self.get_frame_labels(frame_number)
            
            # Добавление визуальных меток на кадр
            if is_start:
                cv2.putText(frame, "START", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Зеленый цвет
            if is_end:
                cv2.putText(frame, "END", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Красный цвет
                
            out.write(frame)  # Запись кадра
            frame_number += 1
            
        # Освобождение ресурсов
        cap.release()
        out.release() 