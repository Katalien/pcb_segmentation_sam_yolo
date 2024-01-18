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
#
# def calculate_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2)
#     union = np.logical_or(mask1, mask2)
#     iou = np.sum(intersection) / np.sum(union)
#     return iou
#
# def calculate_precision_recall(gt_masks, pred_masks, iou_threshold=0.5):
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#
#     for true_mask in gt_masks:
#         iou_scores = [calculate_iou(true_mask, pred_mask) for pred_mask in pred_masks]
#         if any(iou > iou_threshold for iou in iou_scores):
#             true_positives += 1
#         else:
#             false_negatives += 1
#
#     for pred_mask in pred_masks:
#         iou_scores = [calculate_iou(pred_mask, true_mask) for true_mask in gt_masks]
#         if all(iou <= iou_threshold for iou in iou_scores):
#             false_positives += 1
#
#     precision = true_positives / (true_positives + false_positives + 1e-8)
#     recall = true_positives / (true_positives + false_negatives + 1e-8)
#
#     return precision, recall
#
# def calculate_mAP(gt_masks_list, pred_masks_list, iou_threshold=0.5):
#     precisions = []
#     recalls = []
#
#     for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
#         precision, recall = calculate_precision_recall(gt_masks, pred_masks, iou_threshold)
#         precisions.append(precision)
#         recalls.append(recall)
#
#     mean_precision = np.mean(precisions)
#     mean_recall = np.mean(recalls)
#
#     return mean_precision, mean_recall
#
# def val(gt_mask, pred_mask):
#     pass
#
#
# def calculate_metrics(predicted_masks, ground_truth_masks):
#     # Преобразование предсказанных масок и масок истинной правды к бинарному формату
#     threshold = 128  # Произвольный порог для бинаризации изображений
#     binary_predicted_masks = [(mask > threshold).astype(np.uint8) for mask in predicted_masks]
#     binary_ground_truth_masks = [(mask > threshold).astype(np.uint8) for mask in ground_truth_masks]
#
#     # Установка размеров для конкатенации
#     desired_height, desired_width = 480, 640  # Установите желаемые размеры
#
#     # Изменение размера масок для соответствия одинаковым размерам
#     resized_binary_predicted_masks = [cv2.resize(mask, (desired_width, desired_height)) for mask in binary_predicted_masks]
#     resized_binary_ground_truth_masks = [cv2.resize(mask, (desired_width, desired_height)) for mask in binary_ground_truth_masks]
#
#     # Вычисление precision и recall
#     precision = precision_score(np.concatenate(resized_binary_ground_truth_masks),
#                                 np.concatenate(resized_binary_predicted_masks), average='micro')
#     recall = recall_score(np.concatenate(resized_binary_ground_truth_masks),
#                           np.concatenate(resized_binary_predicted_masks), average='micro')
#
#     # Вычисление mAP50 и mAP50-95
#     mAP50 = average_precision_score(np.concatenate(resized_binary_ground_truth_masks), np.concatenate(resized_binary_predicted_masks), average='macro')
#     mAP50_95 = average_precision_score(np.concatenate(resized_binary_ground_truth_masks), np.concatenate(resized_binary_predicted_masks), average='weighted')
#
#     return precision, recall, mAP50, mAP50_95
#
# def calculate_metrics_small(ground_truth_masks, predicted_masks):
#     ground_truth_masks = [cv2.resize(mask, (640, 480)) for mask in ground_truth_masks]
#     predicted_masks = [cv2.resize(mask, (640, 480)) for mask in predicted_masks]
#
#     # Compute IoU (Intersection over Union)
#     intersection = np.logical_and(ground_truth_masks, predicted_masks)
#     union = np.logical_or(ground_truth_masks, predicted_masks)
#     iou = np.sum(intersection) / np.sum(union)
#
#     # Compute True Positives, False Positives, False Negatives
#     true_positives = np.sum(np.logical_and(ground_truth_masks, predicted_masks))
#     false_positives = np.sum(np.logical_and(np.logical_not(ground_truth_masks), predicted_masks))
#     false_negatives = np.sum(np.logical_and(ground_truth_masks, np.logical_not(predicted_masks)))
#
#     # Compute Precision and Recall
#     precision = true_positives / (true_positives + false_positives)
#     recall = true_positives / (true_positives + false_negatives)
#
#     return iou, precision, recall

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