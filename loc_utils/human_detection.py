import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_humans(model, image_path):
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    new_size = (640, 640)
    img_resized = cv2.resize(img_rgb, new_size)
 
    results = model(img_resized)
    
    detections = results.pandas().xyxy[0]

    human_bboxes = detections[detections['name'] == 'person']

    scale_x = original_width / new_size[0]
    scale_y = original_height / new_size[1]
    bboxes = human_bboxes[['xmin', 'ymin', 'xmax', 'ymax']].values * [scale_x, scale_y, scale_x, scale_y]

    return img_rgb, bboxes

def draw_bboxes(image, bboxes):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def crop_human_regions(image, bboxes):
    human_images = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cropped_img = image[ymin:ymax, xmin:xmax]
        human_images.append(cropped_img)

    return human_images