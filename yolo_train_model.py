import os
from ultralytics import YOLO
from PIL import Image
import cv2

### Define paths
# train_images_folder = "train_data/train_images/"
# train_labels_folder = "train_data/train_labels/"
# classes_file = "train_data/classes.txt"
dataset_file = "train_data/custom_dataset.yaml"
# model_weights_file = "model/yolo_custom_model.weights"

model = YOLO('model/yolov8l.yaml')  # build a new model from scratch # n for nano, s for small, m for mid, l for large, x for xlarge
print("Model loaded.")


model.train(data=dataset_file, epochs=100)