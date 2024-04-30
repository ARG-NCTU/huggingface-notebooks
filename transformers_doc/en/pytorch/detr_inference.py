# python3 detr_inference.py -i source_video/WAM_V_1.mp4 -o out_video/WAM_V_1_out.mp4
# python3 detr_inference.py -i source_video/WAM_V_2.mp4 -o out_video/WAM_V_2_out.mp4
# python3 detr_inference.py -i source_video/WAM_V_3.mp4 -o out_video/WAM_V_3_out.mp4
# python3 detr_inference.py -i source_video/Multi_Boat.mp4 -o out_video/Multi_Boat_out.mp4

# python3 detr_inference.py -i source_video/splash1.mp4 -o out_video/splash1_out.mp4
# python3 detr_inference.py -i source_video/splash2.mp4 -o out_video/splash2_out.mp4
# python3 detr_inference.py -i source_video/splash3.mp4 -o out_video/splash3_out.mp4

import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image, ImageDraw
import cv2
import numpy as np
import argparse
import time

# Define class labels and corresponding colors
class_labels = [
    "BallonBoat", "BigBoat", "Boat", "JetSki", "Katamaran",
    "SailBoat", "SmallBoat", "SpeedBoat", "WAM_V"
]
class_colors = {
    "BallonBoat": "yellow", "BigBoat": "green", "Boat": "blue",
    "JetSki": "red", "Katamaran": "purple", "SailBoat": "orange",
    "SmallBoat": "pink", "SpeedBoat": "cyan", "WAM_V": "magenta"
}

# Load the processor and model
image_processor = AutoImageProcessor.from_pretrained("zhuchi76/detr-resnet-50-finetuned-boat-dataset")
model = AutoModelForObjectDetection.from_pretrained("zhuchi76/detr-resnet-50-finetuned-boat-dataset")

# Function to perform detection on a single frame
def detect_objects_in_frame(image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    return results

# Function to draw detections on an image
def draw_detections(image, detections, fps):
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        class_name = model.config.id2label[label.item()]
        box_color = class_colors.get(class_name, "white")  # Default to white if class not found

        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline=box_color, width=2)
        draw.text((x, y), f"{class_name} {score:.2f}", fill=box_color)

    if fps > 0:
        draw.text((10, 10), f"FPS: {fps:.2f}", fill="red")
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='DETR object detection')
    parser.add_argument('-i', '--input', type=str, help='video path, e.g. "input.mp4"')
    parser.add_argument('-o', '--output', type=str, help='output path&name.mp4, e.g. "output.mp4"')
    _args = parser.parse_args()
    return _args

args = parse_args()
input_video_path = args.input
output_video_path = args.output

cap = cv2.VideoCapture(input_video_path)
FPS = cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, FPS, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time_ns()
frame_count = 0
fps = -1  # Initialize the FPS counter

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    image = Image.fromarray(frame[:, :, ::-1])  # Convert the BGR image to RGB

    # Detect objects
    detections = detect_objects_in_frame(image)

    # Calculate FPS
    if frame_count >= 30:  # Update FPS every 30 frames
        end_time = time.time_ns()
        fps = 1000000000 * frame_count / (end_time - start_time)
        frame_count = 0
        start_time = time.time_ns()

    # Draw detections on the image
    image = draw_detections(image, detections, fps)

    # Convert PIL image back to BGR array and write to file
    frame_out = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    out.write(frame_out)  # Write out frame to file

    # cv2.imshow('DETR Detection', frame_out)  # Display the frame
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on Q key
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
