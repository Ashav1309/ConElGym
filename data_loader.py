import cv2
import numpy as np
from typing import Tuple, List
import os
import json
from annotation import VideoAnnotation

class VideoDataLoader:
    def __init__(self, video_path: str, annotation_path: str = None, 
                 target_size: Tuple[int, int] = (224, 224)):
        self.video_path = video_path
        self.target_size = target_size
        self.cap = cv2.VideoCapture(video_path)
        self.annotation = None
        
        if annotation_path and os.path.exists(annotation_path):
            self.annotation = VideoAnnotation.load(annotation_path, video_path)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра"""
        # Изменение размера
        frame = cv2.resize(frame, self.target_size)
        # Нормализация
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def get_frames(self, num_frames: int = None) -> List[np.ndarray]:
        """Получение кадров из видео"""
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if num_frames is not None and frame_count >= num_frames:
                break
                
            # Конвертация в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Предобработка
            frame = self.preprocess_frame(frame)
            frames.append(frame)
            frame_count += 1
            
        self.cap.release()
        return frames
    
    def create_sequences(self, frames: List[np.ndarray], 
                        sequence_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей кадров и меток"""
        sequences = []
        labels = []
        
        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            sequences.append(sequence)
            
            if self.annotation:
                # Получение меток для центрального кадра последовательности
                center_frame = i + sequence_length // 2
                is_start, is_end = self.annotation.get_frame_labels(center_frame)
                labels.append([int(is_start), int(is_end)])
            else:
                labels.append([0, 0])
                
        return np.array(sequences), np.array(labels)
    
    def load_data(self, sequence_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка и подготовка данных"""
        frames = self.get_frames()
        sequences, labels = self.create_sequences(frames, sequence_length)
        return sequences, labels
    
    @classmethod
    def load_dataset(cls, data_dir: str, sequence_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка всего набора данных"""
        all_sequences = []
        all_labels = []
        
        videos_dir = os.path.join(data_dir, 'videos')
        annotations_dir = os.path.join(data_dir, 'annotations')
        
        for video_file in os.listdir(videos_dir):
            if not video_file.endswith(('.mp4', '.avi')):
                continue
                
            video_path = os.path.join(videos_dir, video_file)
            annotation_path = os.path.join(
                annotations_dir,
                f"{os.path.splitext(video_file)[0]}.json"
            )
            
            loader = cls(video_path, annotation_path)
            sequences, labels = loader.load_data(sequence_length)
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            
        return np.concatenate(all_sequences), np.concatenate(all_labels) 