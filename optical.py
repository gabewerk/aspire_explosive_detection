import numpy as np
import os
import cv2
from ultralytics import YOLO

raw_optical_path = "/Users/gabe/Documents/Code/optical_data" # folder path
os.chdir(raw_optical_path) # change the directory
MODEL_NAME = "yolo11n.pt" 
model = YOLO(MODEL_NAME) # load YOLO

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
    if file.endswith(".png"): # check formatting
        file_path = f"{raw_optical_path}/{file}"
        results = model(source=file_path, show=False, save=True, show_labels=False, show_conf=False)
        image = read_image(file_path)
        # display the image using OpenCV's built-in viewer
        #cv2.imshow('Loaded Image', image)
        #cv2.waitKey(0)  # wait until any key is pressed
        #cv2.destroyAllWindows()  # close the window

#trained_model = model.train(dat="coco8.yaml", epochs=100, imgsz=640) # train the model
# for individual images
#results = model(source=raw_optical_path + "/image_2931075238.png", show=True, save=False, conf=0.001, show_labels=False)

