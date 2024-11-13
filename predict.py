from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/train/exp/weights/best.pt")

# Define path to the image file
test_image_Glioma = "dataset/images/val/Tr-gl_1141.jpg"
test_image_Meningioma = "dataset/images/val/Tr-me_1256.jpg"
test_image_NoTumor = "dataset/images/val/Tr-no_0401.jpg"
test_image_Pituitary = "dataset/images/val/Tr-pi_0137.jpg"

# Run inference on the source
results_Glioma = model(test_image_Glioma,save=True)
results_Meningioma = model(test_image_Meningioma,save=True)
results_NoTumor = model(test_image_NoTumor,save=True)
results_Pituitary = model(test_image_Pituitary,save=True)
