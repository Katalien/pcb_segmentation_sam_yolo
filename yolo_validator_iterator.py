import os
import shutil
from ultralytics import YOLO


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}: {e}')


# Пути к папкам
FOLDER_ALL = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/'
TARGET_FOLDER = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid/'

IMAGES_FOLDER = os.path.join(FOLDER_ALL, 'images')
LABELS_FOLDER = os.path.join(FOLDER_ALL, 'labels')

TARGET_IMAGES_FOLDER = os.path.join(TARGET_FOLDER, 'images')
TARGET_LABELS_FOLDER = os.path.join(TARGET_FOLDER, 'labels')

image_files = sorted(os.listdir(IMAGES_FOLDER))
label_files = sorted(os.listdir(LABELS_FOLDER))

model = YOLO('./weights/best_large_val50_150.pt')

for image_file, label_file in zip(image_files, label_files):
    # Получаем текущие содержимые целевых папок
    prev_images = os.listdir(TARGET_IMAGES_FOLDER)
    prev_labels = os.listdir(TARGET_LABELS_FOLDER)

    if prev_images:
        shutil.move(
            os.path.join(TARGET_IMAGES_FOLDER, prev_images[0]),
            os.path.join(IMAGES_FOLDER, prev_images[0])
        )
    if prev_labels:
        shutil.move(
            os.path.join(TARGET_LABELS_FOLDER, prev_labels[0]),
            os.path.join(LABELS_FOLDER, prev_labels[0])
        )

    shutil.move(
        os.path.join(IMAGES_FOLDER, image_file),
        os.path.join(TARGET_IMAGES_FOLDER, image_file)
    )
    shutil.move(
        os.path.join(LABELS_FOLDER, label_file),
        os.path.join(TARGET_LABELS_FOLDER, label_file)
    )

    model.val()