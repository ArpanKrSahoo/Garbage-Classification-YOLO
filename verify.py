import torch
import cv2
import ultralytics

from ultralytics import YOLO

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)

print("CUDA Available:", torch.cuda.is_available())

model = YOLO("my_model/my_model.pt")

print("Model loaded successfully!")
print("Classes:", model.names)