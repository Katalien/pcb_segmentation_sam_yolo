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
            print(f'Ошибка при удалении файла {file_path}. {e}')


folder_all = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/'
target_folder = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid/'
images_folder = folder_all + 'images/'
labels_folder = folder_all + 'labels/'
image_files = os.listdir(images_folder)
labels_files = os.listdir(labels_folder)
image_files.sort()
labels_files.sort()

model = YOLO('./weights/best_large_val50_150.pt')

for image, label in zip(image_files, labels_files):
    target_images_folder = target_folder + 'images/'
    target_labels_folder = target_folder + 'labels/'
    prev_image = os.listdir(target_images_folder)
    prev_label = os.listdir(target_labels_folder)
    if len(prev_image) != 0:
        shutil.move(target_images_folder + prev_image[0], images_folder + prev_image[0])
        shutil.move(target_labels_folder + prev_label[0], labels_folder + prev_label[0])
    shutil.move(images_folder + image, target_images_folder + image)
    shutil.move(labels_folder + label, target_labels_folder + label)
    metrics = model.val()

