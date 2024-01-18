import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

def get_mask_image(masks):
    for mask in masks:
        color = np.array([0, 0, 0, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        print(type(mask_image))
        return mask_image

def show_masks(masks, ax, random_color=False):
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        return mask_image
        # os.makedirs('C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/pred_masks/', exist_ok=True)
        # mask_save_path = os.path.join('C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/pred_masks/', "mask.jpg")
        # mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
        # mask_pil.save(mask_save_path)

def overlay_masks_on_black_background(masks_tensor, save_path=None):
    background = np.zeros(masks_tensor.shape[-2:], dtype=np.uint8)  # Черный фон

    for mask in masks_tensor:
        mask = mask.squeeze().cpu().numpy()  # Преобразование тензора в массив NumPy
        background += (mask * 255).astype(np.uint8)  # Наложение каждой маски на черный фон

    mask_pil = Image.fromarray(background)  # Создание изображения из массива NumPy

    if save_path:
        mask_pil.save(save_path)  # Сохранение изображения
    else:
        mask_pil.show()  # Отображение изображения на экране


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_boxes(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))

def convert(yolo_coords):
    prompt_box = {}
    xc = yolo_coords[0]
    yc = yolo_coords[1]
    width = yolo_coords[2]
    height = yolo_coords[3]
    prompt_box["top_left_x"] = (xc - width/2)
    prompt_box["top_left_y"] = (yc - height/2)
    prompt_box["bottom_right_x"] = (xc + width/2)
    prompt_box["bottom_right_y"] = (yc + height/2)
    return [(xc - width/2)*640, (yc - height/2)*480, (xc + width/2)*640, (yc + height/2)*480]


def get_prompt(model):
    cls = []
    box_prompts = []
    for cl in list(model.predictor.results[0].boxes.xywhn):
        list_cl = list(i.item() for i in cl)
        cls.append(list_cl)
        box_prompts.append(convert(list_cl))
    return box_prompts

def show_raw_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()


