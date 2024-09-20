import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import os
import cv2
import json
import numpy as np
import random
from loc_utils import SimCCLabel
from typing import Tuple
from itertools import product

class SimCCDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        annotation_file: str, 
        dataset_cfg: dict
    ):
        self.data_dir = data_dir
        self.data = self.load_coco_data(annotation_file)

        self.input_shape = (dataset_cfg['input_shape'][0], dataset_cfg['input_shape'][1])
        self.num_joints = dataset_cfg['num_joints']
        self.augment = dataset_cfg['augment']
        self.grayscale = dataset_cfg['grayscale']

        self.split_ratio = dataset_cfg['split_ratio']
        self.sigma = dataset_cfg['sigma']
        self.weights_normalize = dataset_cfg['weights_normalize']

        self.min_max_normalize = dataset_cfg['min_max_normalize']
        self.min_max_scale = dataset_cfg['min_max_scale']

        self.encoder = SimCCLabel(input_size=self.input_shape, 
                                  sigma=self.sigma, 
                                  split_ratio=self.split_ratio, 
                                  normalize=self.weights_normalize,
                                  min_max_normalize=self.min_max_normalize,
                                  min_max_scale=self.min_max_scale)

        self.symmetric_pairs = {
            1: 2, 3: 4, 5: 6,      # Left-right eyes, ears, shoulders
            7: 8, 9: 10, 11: 12,   # Left-right elbows, wrists, hips
            13: 14, 15: 16         # Left-right knees, ankles
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        item = self.data[index]
        img_path = os.path.join(self.data_dir, item['file_name'])
        cropped_images, keypoints, encoded = self._process_image(img_path, item)
        return cropped_images, keypoints, encoded

    def _process_image(
        self, 
        img_path: str,
        item: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        keypoints = np.array(item['keypoints']).reshape(-1, 17, 3).astype(np.float32)

        cropped_images = []
        cropped_keypoints = []
        cropped_visibility = []
        
        for bbox, kps in zip(item['bboxes'], keypoints):
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cropped_image = image[y1:y2, x1:x2]

            # Scale keypoints to cropped image
            kps[:, 0] -= x1
            kps[:, 1] -= y1

            if self.augment:
                cropped_image, kps = self.augment_image_and_kps(cropped_image, kps)

            kps[:, 0] *= self.input_shape[0] / cropped_image.shape[1]
            kps[:, 1] *= self.input_shape[1] / cropped_image.shape[0]

            cropped_keypoints.append(kps[:, :2].copy())

            visibility = (kps[:, 2] == 2).astype(np.float32)

            cropped_image = cv2.resize(cropped_image, (self.input_shape[0], self.input_shape[1]))
            cropped_image = cropped_image / 255.0

            if self.grayscale:
                cropped_image = np.expand_dims(cropped_image, axis=-1)

            cropped_images.append(cropped_image)
            cropped_visibility.append(visibility)

        encoded = self.encoder.encode(np.array(cropped_keypoints), np.array(cropped_visibility))

        cropped_images = np.array(cropped_images).transpose(0, 3, 1, 2)  # Convert to NCHW format
        cropped_images = torch.tensor(cropped_images).float()
        cropped_keypoints = torch.tensor(np.array(cropped_keypoints)).float()

        return cropped_images, cropped_keypoints, encoded

    def augment_image_and_kps(
        self, 
        image: np.ndarray,
        kps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = np.array(image).transpose(2, 0, 1)
        brightness_factor = np.random.uniform(0.7, 1.3)
        image = F.adjust_brightness(torch.tensor(image), brightness_factor=brightness_factor)
        saturation_factor = np.random.uniform(0.7, 1.3)
        image = F.adjust_saturation(image, saturation_factor=saturation_factor)

        image = np.array(image).transpose(1, 2, 0)
        angle = np.random.uniform(-20, 20)
        image, kps = self.rotate_image_and_keypoints(image, kps, angle)

        if np.random.rand() > 0.5:
            image, kps = self.flip_image_and_keypoints(image, kps)

        return image, kps

    def rotate_image_and_keypoints(
        self, 
        image: np.ndarray,
        keypoints: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        ones = np.ones(shape=(len(keypoints), 1))
        keypoints_homogeneous = np.hstack([keypoints[:, :2], ones])
        rotated_keypoints = rotation_matrix.dot(keypoints_homogeneous.T).T

        visibility = keypoints[:, 2].reshape(-1, 1)
        rotated_keypoints = np.hstack([rotated_keypoints, visibility])

        return rotated_image, rotated_keypoints

    def flip_image_and_keypoints(
        self, 
        img: np.ndarray,
        keypoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        flipped_img = cv2.flip(img, 1)

        keypoints[:, 0] = w - keypoints[:, 0] - 1

        for left, right in self.symmetric_pairs.items():
            keypoints[left, :], keypoints[right, :] = keypoints[right, :].copy(), keypoints[left, :].copy()

        return flipped_img, keypoints

    def load_coco_data(
        self, 
        annotation_file: str
    ) -> list:
        with open(annotation_file) as f:
            annotations = json.load(f)
        
        images = {image_info['id']: image_info['file_name'] for image_info in annotations['images']}
        
        data = []
        for ann in annotations['annotations']:
            if ann['num_keypoints'] > 0:
                image_id = ann['image_id']
                image_name = images.get(image_id)

                if image_name is None or not os.path.exists(os.path.join(self.data_dir, image_name)):
                    continue

                bbox = ann['bbox']
                keypoints = ann['keypoints']

                data.append({
                    'file_name': image_name,
                    'bboxes': [bbox],
                    'keypoints': [keypoints],
                    'visibility': [keypoints[2::3]]
                })

        return data