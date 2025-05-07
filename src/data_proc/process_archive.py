import os
import zipfile
from pathlib import Path
import random
import shutil

def process_archive(archive_path, output_dir):
    """
    Обрабатывает архив с видео, разделяя файлы по папкам на основе первой части имени файла.
    Работает напрямую с архивом без распаковки во временной директории.
    
    Args:
        archive_path (str): Путь к архиву
        output_dir (str): Директория для сохранения результатов
    """
    # Создаем директорию data, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Открываем архив
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Получаем список всех файлов в архиве
            file_list = zip_ref.namelist()
            
            # Фильтруем только видео файлы из папки videos
            video_files = [f for f in file_list if f.startswith('videos/') and f.lower().endswith('.mp4')]
            
            for video_path in video_files:
                # Получаем имя файла из пути
                file_name = os.path.basename(video_path)
                
                # Разделяем имя файла по '_' и берем первую часть
                folder_name = file_name.split('_')[0]
                
                # Создаем целевую директорию
                target_dir = os.path.join(output_dir, folder_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # Проверяем, существует ли уже такой файл
                target_file = os.path.join(target_dir, file_name)
                if os.path.exists(target_file):
                    print(f"Файл уже существует, пропускаем: {file_name}")
                    continue
                
                # Извлекаем файл напрямую в целевую директорию
                zip_ref.extract(video_path, target_dir)
                
                # Переименовываем файл, убирая путь videos/
                old_path = os.path.join(target_dir, video_path)
                new_path = os.path.join(target_dir, file_name)
                os.rename(old_path, new_path)
                
                # Удаляем пустую папку videos
                videos_dir = os.path.join(target_dir, 'videos')
                if os.path.exists(videos_dir):
                    os.rmdir(videos_dir)
                
                print(f"Обработан файл: {file_name} -> {folder_name}")
        
        print("Обработка архива завершена успешно!")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

def split_train_test(data_dir, train_count=150, val_count=50):
    """
    Распределяет видеофайлы между папками train, validation и test.
    Из каждой папки с элементами берет указанное количество видео для train и validation.
    
    Args:
        data_dir (str): Директория с данными
        train_count (int): Количество видео для train из каждой папки
        val_count (int): Количество видео для validation из каждой папки
    """
    # Создаем директории train, validation и test
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Получаем список всех папок в data_dir
        element_folders = [f for f in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, f)) 
                         and f not in ['train', 'validation', 'test']]
        
        for element_folder in element_folders:
            element_path = os.path.join(data_dir, element_folder)
            
            # Получаем список всех видеофайлов в папке
            video_files = [f for f in os.listdir(element_path) if f.lower().endswith('.mp4')]
            
            # Проверяем, достаточно ли файлов
            total_needed = train_count + val_count
            if len(video_files) < total_needed:
                print(f"Предупреждение: В папке {element_folder} меньше {total_needed} видео")
                continue
            
            # Перемешиваем список файлов
            random.shuffle(video_files)
            
            # Выбираем файлы для train
            train_files = video_files[:train_count]
            
            # Выбираем файлы для validation
            val_files = video_files[train_count:train_count + val_count]
            
            # Оставшиеся файлы идут в test
            test_files = video_files[train_count + val_count:]
            
            # Перемещаем файлы в train
            for file in train_files:
                src = os.path.join(element_path, file)
                dst = os.path.join(train_dir, file)
                shutil.move(src, dst)
                print(f"Перемещен файл в train: {file}")
            
            # Перемещаем файлы в validation
            for file in val_files:
                src = os.path.join(element_path, file)
                dst = os.path.join(val_dir, file)
                shutil.move(src, dst)
                print(f"Перемещен файл в validation: {file}")
            
            # Перемещаем файлы в test
            for file in test_files:
                src = os.path.join(element_path, file)
                dst = os.path.join(test_dir, file)
                shutil.move(src, dst)
                print(f"Перемещен файл в test: {file}")
            
            # Удаляем пустую папку элемента
            if not os.listdir(element_path):
                os.rmdir(element_path)
            
            print(f"Обработана папка: {element_folder}")
        
        print("Распределение файлов между train, validation и test завершено успешно!")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    archive_path = r"C:\Users\Admin\Downloads\RG_public.zip"
    output_dir = "data"
    
    # Обрабатываем архив
    #process_archive(archive_path, output_dir)
    
    # Распределяем файлы между train, validation и test
    split_train_test(output_dir) 