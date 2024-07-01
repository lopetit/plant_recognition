import cv2
import os
import argparse

# Define paths

image_folder = "train_data/train_images/"
annotation_folder = "train_data/train_labels/"
output_folder = "train_data/train_images_processed/"
classes_file = "train_data/classes.txt"

# Load classes
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

train_images_number = len(os.listdir(image_folder))

# Process each image and annotation file
for i in range(train_images_number):  # Assuming you have test0.jpg to test(max_number-1).jpg
    image_path = os.path.join(image_folder, f"train{i}.jpg")
    annotation_path = os.path.join(annotation_folder, f"train{i}.txt")

    # Load image
    image = cv2.imread(image_path)

    # Read annotation file
    with open(annotation_path, "r") as f:
        annotations = [line.strip() for line in f.readlines()]

    # Add bounding boxes and labels to the image
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())

        # Calculate box coordinates
        x = int((x_center - width / 2) * image.shape[1])
        y = int((y_center - height / 2) * image.shape[0])
        w = int(width * image.shape[1])
        h = int(height * image.shape[0])

        # Draw black rectangle background for the label
        label = f"{classes[int(class_id)]}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x, y - label_height - 10), (x + label_width, y - 5), (0, 0, 0), cv2.FILLED)

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save processed image
    output_path = os.path.join(output_folder, f"processed_train{i}.jpg")
    cv2.imwrite(output_path, image)

    print(f"Processed {image_path} -> Saved {output_path}")

print("Processing completed.")
