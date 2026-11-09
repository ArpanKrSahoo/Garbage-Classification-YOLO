# Garbage Detection and Disposal Recommendation using YOLO + OpenCV
# Install dependencies:
# pip install ultralytics opencv-python

import os
import sys
from collections import Counter

import cv2
from ultralytics import YOLO


# -------------------- CONFIGURATION --------------------

# Update this to the path of your trained garbage detection model.
# Example: "runs/detect/train/weights/best.pt"
model_path = "my_model/my_model.pt"

min_thresh = 0.50
cam_index = 0
imgW, imgH = 1280, 720
record = False

# These class names should match your data.yaml.
# The model's own class names are loaded below and used for predictions.
disposal_info = {
    "BIODEGRADABLE": (
        "Place in the wet/organic waste bin. Compost suitable food and garden waste."
    ),
    "CARDBOARD": (
        "Keep dry, flatten it, and place it in the paper/cardboard recycling bin."
    ),
    "GLASS": (
        "Place in the designated glass recycling container. Handle broken glass carefully "
        "and follow your campus collection rules."
    ),
    "METAL": (
        "Place in the metal recycling bin. Empty containers before disposal."
    ),
    "PAPER": (
        "Keep paper clean and dry, then place it in the paper recycling bin."
    ),
    "PLASTIC": (
        "Empty and rinse recyclable plastic containers, then use the designated plastic "
        "recycling bin. Check local recycling rules."
    ),
}

# Fallback advice for an unexpected class name in the trained model.
DEFAULT_ADVICE = (
    "Check the label and follow your campus waste-segregation guidelines."
)

bbox_colors = [
    (60, 180, 75),
    (230, 159, 0),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
]


# -------------------- LOAD MODEL --------------------

if not os.path.isfile(model_path):
    print(f"ERROR: Model file not found: {model_path}")
    print("Set model_path to your trained YOLO best.pt file.")
    sys.exit(1)

model = YOLO(model_path, task="detect")
labels = model.names

print("Model loaded successfully.")
print("Model classes:", labels)


# -------------------- CAMERA SETUP --------------------

cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("ERROR: Could not open the camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgH)

recorder = None
if record:
    record_name = "garbage_detection.avi"
    record_fps = 20
    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*"MJPG"),
        record_fps,
        (imgW, imgH),
    )


# -------------------- REAL-TIME DETECTION LOOP --------------------

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("ERROR: Unable to read from the camera.")
            break

        # Run YOLO object detection.
        # Tracking is not required for basic waste classification/detection.
        results = model.predict(
            source=frame,
            conf=min_thresh,
            verbose=False
        )

        detections = results[0].boxes
        detected_classes = []

        for detection in detections:
            xyxy = detection.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            class_id = int(detection.cls.item())
            confidence = float(detection.conf.item())
            class_name = str(labels[class_id]).upper()

            detected_classes.append(class_name)

            color = bbox_colors[class_id % len(bbox_colors)]

            # Draw bounding box.
            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmax, ymax),
                color,
                2
            )

            # Draw class name and confidence.
            label_text = f"{class_name}: {confidence * 100:.1f}%"
            text_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            label_y = max(ymin, text_size[1] + 10)
            cv2.rectangle(
                frame,
                (xmin, label_y - text_size[1] - 8),
                (xmin + text_size[0] + 6, label_y + baseline - 5),
                color,
                cv2.FILLED
            )
            cv2.putText(
                frame,
                label_text,
                (xmin + 3, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

        # Count detected waste objects in the current frame.
        class_counts = Counter(detected_classes)
        total_objects = len(detected_classes)

        # Draw a semi-transparent information panel.
        overlay = frame.copy()
        panel_height = min(190, 55 + len(class_counts) * 27)
        cv2.rectangle(
            overlay,
            (10, 10),
            (610, panel_height),
            (25, 25, 25),
            cv2.FILLED
        )
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

        cv2.putText(
            frame,
            f"TOTAL WASTE ITEMS: {total_objects}",
            (22, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        y_position = 72
        for class_name, count in class_counts.most_common():
            line = f"{class_name}: {count}"
            cv2.putText(
                frame,
                line,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (120, 230, 160),
                2
            )
            y_position += 27

        # Show disposal advice for the most confident detection.
        if len(detections) > 0:
            best_detection_index = int(
                detections.conf.argmax().item()
            )
            best_class_id = int(
                detections[best_detection_index].cls.item()
            )
            best_class_name = str(labels[best_class_id]).upper()
            advice = disposal_info.get(best_class_name, DEFAULT_ADVICE)

            # Keep the advice panel readable by wrapping it into short lines.
            cv2.rectangle(
                frame,
                (10, imgH - 105),
                (imgW - 10, imgH - 10),
                (35, 35, 35),
                cv2.FILLED
            )
            cv2.putText(
                frame,
                f"DISPOSAL: {best_class_name}",
                (22, imgH - 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2
            )

            words = advice.split()
            lines = []
            current_line = ""
            max_chars = 105

            for word in words:
                test_line = (current_line + " " + word).strip()
                if len(test_line) > max_chars and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            for index, line in enumerate(lines[:2]):
                cv2.putText(
                    frame,
                    line,
                    (22, imgH - 43 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1
                )
        else:
            cv2.putText(
                frame,
                "No waste object detected",
                (22, imgH - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.imshow("Garbage Classification and Disposal Assistant", frame)

        if recorder is not None:
            recorder.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            cv2.waitKey(0)  # Press any key to resume.
        elif key in (ord("p"), ord("P")):
            cv2.imwrite("garbage_detection_capture.jpg", frame)
            print("Saved: garbage_detection_capture.jpg")

finally:
    cap.release()
    if recorder is not None:
        recorder.release()
    cv2.destroyAllWindows()
