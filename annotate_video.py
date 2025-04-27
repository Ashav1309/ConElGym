import cv2
import os
from annotation import VideoAnnotation
from config import Config

class VideoAnnotator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.annotation = VideoAnnotation(video_path)
        self.current_frame = 0
        self.paused = False
        self.start_frame = None
        
    def process_frame(self, frame):
        """Обработка и отображение кадра"""
        # Отображение номера текущего кадра
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Отображение инструкций
        cv2.putText(frame, "Space: pause/resume", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "S: mark start", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "E: mark end", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Q: quit", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame = self.process_frame(frame)
                cv2.imshow('Video Annotation', frame)
                self.current_frame += 1
                
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Пауза/продолжение
                self.paused = not self.paused
            elif key == ord('s'):  # Отметить начало
                self.start_frame = self.current_frame
                print(f"Start frame marked: {self.start_frame}")
            elif key == ord('e'):  # Отметить конец
                if self.start_frame is not None:
                    self.annotation.add_element(
                        self.start_frame,
                        self.current_frame,
                        "element"  # Здесь можно указать тип элемента
                    )
                    print(f"Element marked: {self.start_frame} - {self.current_frame}")
                    self.start_frame = None
            elif key == ord('q'):  # Выход
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Сохранение разметки
        video_name = os.path.basename(self.video_path)
        annotation_path = os.path.join(
            Config.TRAIN_DATA_PATH,
            'annotations',
            f"{os.path.splitext(video_name)[0]}.json"
        )
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        self.annotation.save(annotation_path)
        
        # Визуализация разметки
        output_path = os.path.join(
            Config.TRAIN_DATA_PATH,
            'processed',
            f"annotated_{video_name}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.annotation.visualize(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python annotate_video.py <video_path>")
        sys.exit(1)
        
    annotator = VideoAnnotator(sys.argv[1])
    annotator.run() 