import cv2
import os
from annotation import VideoAnnotation
from config import Config

class VideoAnnotator:
    def __init__(self, video_path: str):
        """
        Инициализация класса для аннотирования видео.
        
        Args:
            video_path (str): Путь к видеофайлу для аннотирования
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)  # Открываем видеофайл
        self.annotation = VideoAnnotation(video_path)  # Создаем объект для хранения аннотаций
        self.current_frame = 0  # Текущий кадр
        self.paused = False  # Флаг паузы
        self.start_frame = None  # Кадр начала элемента
        
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
        cv2.putText(frame, "Q: quit", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """
        Основной цикл аннотирования видео.
        Обрабатывает видео кадр за кадром, позволяет отмечать начало и конец элементов.
        """
        while True:
            # Если видео не на паузе, читаем следующий кадр
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:  # Если видео закончилось
                    break
                    
                frame = self.process_frame(frame)  # Обрабатываем кадр
                cv2.imshow('Video Annotation', frame)  # Отображаем кадр
                self.current_frame += 1  # Увеличиваем счетчик кадров
                
            # Обработка нажатий клавиш
            key = cv2.waitKey(1) & 0xFF
            
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
            elif key == ord('q'):  # Выход из программы
                break
                
        # Освобождаем ресурсы
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Сохранение разметки в JSON файл
        video_name = os.path.basename(self.video_path)
        annotation_path = os.path.join(
            Config.TRAIN_DATA_PATH,
            'annotations',
            f"{os.path.splitext(video_name)[0]}.json"
        )
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        self.annotation.save(annotation_path)
        
        # Визуализация разметки на видео
        output_path = os.path.join(
            Config.TRAIN_DATA_PATH,
            'processed',
            f"annotated_{video_name}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.annotation.visualize(output_path)

if __name__ == "__main__":
    import sys
    # Проверка аргументов командной строки
    if len(sys.argv) != 2:
        print("Usage: python annotate_video.py <video_path>")
        sys.exit(1)
        
    # Создание и запуск аннотатора
    annotator = VideoAnnotator(sys.argv[1])
    annotator.run() 