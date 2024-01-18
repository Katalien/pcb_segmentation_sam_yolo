from helper import *
from mobile_sam import sam_model_registry, SamPredictor
from ultralytics import YOLO, SAM
import os
import count_area

def create_yolo():
    model = YOLO('./weights/best_nano_val50.pt')
    return model

def predict_yolo(img_path, model):
    model.predict(img_path, imgsz=640, conf=0.4)
    return model


def create_mobile_sam():
    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def predict_yolo(img_path, model):
    model.predict(img_path, imgsz=640, conf=0.5, save_conf=True)
    return model

def run_predict(image_path, model, predictor, show=False, save_path=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.path.append("..")
    predictor.set_image(image)
    predicted_yolo = predict_yolo(image_path, model)
    _boxes = get_prompt(predicted_yolo)
    input_boxes = torch.tensor(_boxes, device='cpu')
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    masks, iou_predictions, low_res_masks = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    mask_image = get_mask_image(masks)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        mask_image = show_masks(masks, plt.gca())
        show_boxes(_boxes, plt.gca())
        plt.axis('off')
        plt.show()
    if save_path:
        save_path = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/res_masks/res.jpg'
        overlay_masks_on_black_background(masks, save_path)
    return mask_image, masks, iou_predictions, low_res_masks


def main():
    model = create_yolo()
    predictor = create_mobile_sam()

    folder_all = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/'
    target_folder = 'C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid/'
    images_folder = folder_all + 'images/'
    labels_folder = folder_all + 'labels/'
    image_files = os.listdir(images_folder)
    labels_files = os.listdir(labels_folder)
    image_files.sort()
    labels_files.sort()

    for image, label in zip(image_files, labels_files):
        mask_image, masks, iou_predictions, low_res_masks = run_predict(images_folder + image, model, predictor, show=True, save_path=True)
        _image_path = "C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/res_masks/"
        count_area.calculate_area_and_perimeter(image, _image_path + 'res.jpg')



if __name__ == "__main__":
     main()