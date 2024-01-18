import cv2
import numpy as np

def calculate_area_and_perimeter(image_name, image_path, scale_factor=0.1, min_contour_area=50):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if image is None:
        print(f"Error: Unable to load the image {image_path}")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    print("Number of contours after filtering:", len(filtered_contours))

    write_results_to_file(image_name, filtered_contours)
    return

def write_results_to_file(image_name, contours, file_path='area_res.txt'):
    with open(file_path, 'a') as file:
        file.write(f"\n{image_name}:\n")
        for i, contour in enumerate(contours, start=1):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            file.write(f"{i}. Square: {area}\n")
            file.write(f"Perimeter: {perimeter}\n")

# # Example usage
# image_path = "C:/Users/z.kate/source/GitHubRepos/ELC_Fall_git/YOLO_SAM/datasets/valid_all/res_masks/WIN_20221023_15_16_07_Pro_jpg.rf.8a0c8bc3af242d3f85123e89ed585656.jpg"
# name = "WIN_20221023_15_16_07_Pro_jpg.rf.8a0c8bc3af242d3f85123e89ed585656.jpg"
# scale_factor = 0.1
# min_contour_area = 50  # Set your minimum contour area threshold
# calculate_area_and_perimeter(name, image_path)
