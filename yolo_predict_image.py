import os
from ultralytics import YOLO
from PIL import Image
import cv2

# Define paths
# predict_images_folder = "train_data/train_images"
predict_images_folder = "test_images/"
# model_weights_file = "model/yolo_custom_model.weights"


# model = YOLO('yolov8n.yaml')
model = YOLO('model/custom_1.pt')
print("Model loaded.")

predict_images_number = len(os.listdir(predict_images_folder))

for i in range(predict_images_number):  # Assuming you have test0.jpg to test(max_number-1).jpg
    image_path = os.path.join(predict_images_folder, f"train{i}.jpg")
    # annotation_path = os.path.join(annotation_folder, f"train{i}.txt")
    im1 = Image.open(image_path)
    results = model.predict(source=im1, save=True)  # save plotted images