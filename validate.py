import os
from sklearn.metrics import average_precision_score
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import helper
from sklearn.metrics import precision_score, recall_score, average_precision_score
from yolo_mobile_sam import *

def read_masks_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        masks = []
        for line in lines:
            values = line.split()[1:]
            print(values)
            mask_coords = [float(value) * 640 if i % 2 == 0 else float(value) * 480 for i, value in enumerate(values)]
            print(mask_coords)
            mask_coords = np.array(mask_coords).reshape(-1, 2)
            masks.append(mask_coords)
    return masks
    
def compute_iou(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def compute_precision_recall(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    precision = np.sum(intersection) / (np.sum(predicted_mask) + 1e-10)
    recall = np.sum(intersection) / (np.sum(ground_truth_mask) + 1e-10)
    return precision, recall

def compute_mAP(ground_truth_masks, predicted_masks):
    # Преобразуйте маски в 1D массивы (необходимо для average_precision_score)
    ground_truth = ground_truth_masks.ravel()
    predicted = predicted_masks.ravel()

    # Вычислите Average Precision
    mAP = average_precision_score(ground_truth, predicted)
    return mAP

def calculate_metrics(gt_mask, pred_mask):

    # Преобразование в двоичные значения
    gt_binary = (gt_mask > 0).astype(int)
    pred_binary = (pred_mask > 0).astype(int)

    # Рассчет precision и recall
    precision = precision_score(gt_binary.ravel(), pred_binary.ravel())
    recall = recall_score(gt_binary.ravel(), pred_binary.ravel())

    # Рассчет mAP
    mAP = average_precision_score(gt_binary.ravel(), pred_binary.ravel())

    return precision, recall, mAP

def main():
    precisions, recalls, mAPs, mAPs95 = [], [], [], []
    folder = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/'
    masks_folder = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/masks/'
    pred_masks_folder = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/pred_masks/'
    masks_files = os.listdir(masks_folder)
    pred_masks_files = os.listdir(pred_masks_folder)
    masks_files.sort()
    pred_masks_files.sort()

    for gt_mask_name, pred_mask_name in zip(masks_files, pred_masks_files):
        gt_mask_path = masks_folder + gt_mask_name
        pred_mask_path = pred_masks_folder + pred_mask_name
        predicted_masks = []
        ground_truth_masks = []
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_masks.append(gt_mask)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_masks.append(pred_mask)
        # Установка размеров для конкатенации
        desired_height, desired_width = 480, 640  # Установите желаемые размеры

        # Изменение размера масок для соответствия одинаковым размерам
        gt_mask = cv2.resize(gt_mask, (desired_width, desired_height))
        pred_mask = cv2.resize(pred_mask, (desired_width, desired_height))

        precision, recall, mAP = calculate_metrics(gt_mask, pred_mask)

        precisions.append(precision)
        recalls.append(recall)
        mAPs.append(mAP)
        # mAPs95.append(mAP50_95)

        # print(f'Precision: {precision}')
        # print(f'Recall: {recall}')
        # print(f'mAP50: {mAP50}')
        # print(f'mAP50-95: {mAP50_95}')
        # print()

    print("precision")
    print(*precisions, sep="\n")
    print("recall")
    print(*recalls, sep='\n')

    fig, ax = plt.subplots(figsize=(8, 6))
    box = plt.boxplot(recalls, widths=0.6, patch_artist=True)
    for whisker in box['whiskers']:
        whisker.set(color='green', linewidth=2)  # Линии "усов"
    for cap in box['caps']:
        cap.set(color='blue', linewidth=2)  # Линии концов "усов"
    for median in box['medians']:
        median.set(color='red', linewidth=2)  # Линии медианы

    plt.title('Recall')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
