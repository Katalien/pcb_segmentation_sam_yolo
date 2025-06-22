import cv2
import numpy as np

def calculate_area_and_perimeter(image_name, image_path, scale_factor=0.1, min_contour_area=50):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load the image {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

