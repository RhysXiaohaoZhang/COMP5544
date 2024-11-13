# Function to load images and labels
import os

import cv2

data_path = 'dataset'

def load_data(data_path, dataType):
    images = []
    labels = []
    image_del = []
    label_del = []

    image_path = os.path.join(data_path, 'images', dataType)
    label_path = os.path.join(data_path, 'labels', dataType)
    for img_file in os.listdir(image_path):
        image_file_path = os.path.join(image_path, img_file)
        img = cv2.imread(os.path.join(image_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_file = img_file.replace('.jpg', '.txt')
        label_file_path = os.path.join(label_path, label_file)
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                label_data = file.readline().strip().split()
                if len(label_data) > 0:
                    images.append(img)
                    labels.append(label_data)
                else:
                    print(f"Label file {label_file_path} is empty, skipping this image.")
                    label_del.append(label_file_path)
        else:
            print(f"Label file {label_file_path} not found, skipping this image.")
            image_del.append(image_file_path)
    return image_del, label_del


image_delT, label_delT = load_data(data_path, 'train')
image_delV, label_delV = load_data(data_path, 'val')

for path in image_delT:
    os.remove(path)
for path in label_delT:
    os.remove(path)
for path in image_delV:
    os.remove(path)
for path in label_delV:
    os.remove(path)