import numpy as np
import os
import cv2
from ultralytics import YOLO

raw_optical_path = "/Users/werk/Documents/Code/aspire_explosive_detection/optical_sensor/optical_data" # folder path
os.chdir(raw_optical_path) # change the directory

def read_image(input):
    image = cv2.imread(input)
    # check if the image was successfully loaded
    if image is None:
        print("Error: Could not open or find the image.")
    else:
        print("Image loaded.")
        return image

# reads the images from the directory
for file in os.listdir():
    if file.endswith(".jpeg"): # check formatting
        file_path = f"{raw_optical_path}/{file}"
        image = read_image(file_path)

# object detection
MODEL_NAME = "yolo11n.pt"
# load YOLO
model = YOLO(MODEL_NAME)
# train the model
#results = model.train(dat="coco8.yaml", epochs=100, imgsz=640)
results = model(source=raw_optical_path + "/image.jpeg")

print(f"Shape: {image.shape}")  # print the shape of image. (height, width, color channels)
#print(f"Pixel [0,0]: {image[0,0]}") # print the contents of pixel at [0,0]

# display the image using OpenCV's built-in viewer
cv2.imshow('Loaded Image', image)
cv2.waitKey(0)  # wait until any key is pressed
cv2.destroyAllWindows()  # close the window