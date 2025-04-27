import json
import os
from typing import List, Dict, Tuple
import cv2

class VideoAnnotation:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.annotations = []
        self.video_name = os.path.basename(video_path)
        
    def add_element(self, start_frame: int, end_frame: int, 
                   element_type: str, confidence: float = 1.0):
        """Добавление разметки элемента"""
        annotation = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'element_type': element_type,
            'confidence': confidence
        }
        self.annotations.append(annotation)
        
    def save(self, output_path: str):
        """Сохранение разметки в JSON файл"""
        data = {
            'video_name': self.video_name,
            'annotations': self.annotations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    @classmethod
    def load(cls, annotation_path: str, video_path: str):
        """Загрузка разметки из JSON файла"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        annotation = cls(video_path)
        annotation.annotations = data['annotations']
        return annotation
    
    def get_frame_labels(self, frame_number: int) -> Tuple[bool, bool]:
        """Получение меток для конкретного кадра"""
        is_start = False
        is_end = False
        
        for ann in self.annotations:
            if frame_number == ann['start_frame']:
                is_start = True
            if frame_number == ann['end_frame']:
                is_end = True
                
        return is_start, is_end
    
    def visualize(self, output_path: str):
        """Визуализация разметки на видео"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            is_start, is_end = self.get_frame_labels(frame_number)
            
            # Добавление визуальных меток
            if is_start:
                cv2.putText(frame, "START", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_end:
                cv2.putText(frame, "END", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            out.write(frame)
            frame_number += 1
            
        cap.release()
        out.release() 