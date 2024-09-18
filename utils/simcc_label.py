import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import scipy.ndimage
from typing import Tuple, Optional
from itertools import product
import cv2

# Can be used for encoding labled joints and decode predicted labels
class SimCCLabel:
    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: float = 6.0,
        split_ratio: float = 2.0,
        label_smooth_weight: float = 0.0,
        normalize: bool = True,
        min_max_normalize: bool = False,
        min_max_scale: float = 25.0,
        decode_beta: float = 150.0,
        decode_visibility: bool = False,
    ):
        super().__init__() 

        self.input_size = input_size
        self.split_ratio = split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize
        self.min_max_normalize = min_max_normalize
        self.min_max_scale = min_max_scale
        self.decode_beta = decode_beta
        self.decode_visibility = decode_visibility

        self.sigma = np.array([sigma, sigma])
    
    def encode(self,
               keypoints: np.ndarray,
               keypoints_visibility: Optional[np.ndarray] = None) -> dict:

        if keypoints_visibility is None:
            keypoints_visibility = np.ones(keypoints.shape[:2], dtype=np.float32)
        
        x_labels, y_labels, keypoint_weights = self._generate_gaussian(
            keypoints, keypoints_visibility) 

        if self.min_max_normalize:
            x_labels *= self.min_max_scale
            y_labels *= self.min_max_scale
        
        encoded = dict(
            keypoints_x_labels = x_labels,
            keypoints_y_labels = y_labels,
            keypoint_weights = keypoint_weights
        )

        return encoded
    
    def decode(self,
               simcc_x: np.ndarray,
               simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        

        keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y)

        # DARK refinement
        x_blur = int((self.sigma[0] * 20 - 7) // 3)
        y_blur = int((self.sigma[1] * 20 - 7) // 3)
        x_blur -= int((x_blur % 2) == 0)
        y_blur -= int((y_blur % 2) == 0)
        keypoints[:, :, 0] = self._refine_simcc_dark(keypoints[:, :, 0], simcc_x, x_blur)
        keypoints[:, :, 1] = self._refine_simcc_dark(keypoints[:, :, 1], simcc_y, y_blur)

        keypoints /= self.split_ratio
        
        if self.decode_visibility:
            _, visibility = self._get_simcc_maximum(
                simcc_x * self.decode_beta * self.sigma[0],
                simcc_y * self.decode_beta * self.sigma[1],
                apply_softmax=True)
            
            return keypoints, (scores, visibility)
        else:
            return keypoints, scores
    
    def _map_coordinates(
        self, 
        keypoints: np.ndarray,
        keypoints_visibility: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = keypoints_visibility.copy()

        return keypoints_split, keypoint_weights

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visibility: Optional[np.ndarray] = None 
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.split_ratio).astype(int)
        H = np.around(h * self.split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visibility)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visibility[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * self.sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * self.sigma[1]**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]

        return target_x, target_y, keypoint_weights

    def _get_simcc_maximum(self,
                           simcc_x: np.ndarray,
                           simcc_y: np.ndarray,
                           apply_softmax: bool = False
                           ) -> Tuple[np.ndarray, np.ndarray]:
        
        N, K, Wx = simcc_x.shape 
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        if apply_softmax: 
            simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
            simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
            ex, ey = np.exp(simcc_x), np.exp(simcc_y)
            simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
            simcc_y = ey / np.sum(ey, axis=1, keepdims=True)
        
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

        max_vals_x = np.amax(simcc_x, axis=1)
        max_vals_y = np.amax(simcc_y, axis=1)

        mask = max_vals_x > max_vals_y
        max_vals_x[mask] = max_vals_y[mask]
        vals = max_vals_x
        locs[vals <= 0.0] = -1

        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
        
        return locs, vals

    def _refine_simcc_dark(self,
                           keypoints: np.ndarray,
                           simcc: np.ndarray,
                           blur_kernel_size: int) -> np.ndarray:
        N, K, Wx = simcc.shape
        
        # Gaussian blur 1d
        border = (blur_kernel_size - 1) // 2
        for n, k in product(range(N), range(K)):
            origin_max = np.max(simcc[n, k]) 
            dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
            dr[0, border:-border] = simcc[n, k].copy()
            dr = cv2.GaussianBlur(dr, (blur_kernel_size, 1), 0)
            simcc[n, k] = dr[0, border:-border].copy()
            simcc[n, k] *= origin_max / np.max(simcc[n, k])
        
        np.clip(simcc, 1e-3, 50., simcc)
        np.log(simcc, simcc)

        simcc = np.pad(simcc, ((0, 0), (0, 0), (2, 2)), 'edge')

        for n in range(N):
            px = (keypoints[n] + 2.5).astype(np.int64).reshape(-1, 1)

            dx0 = np.take_along_axis(simcc[n], px, axis=1)
            dx1 = np.take_along_axis(simcc[n], px + 1, axis=1)
            dx_1 = np.take_along_axis(simcc[n], px - 1, axis=1)
            dx2 = np.take_along_axis(simcc[n], px + 2, axis=1)
            dx_2 = np.take_along_axis(simcc[n], px - 2, axis=1)

            dx = 0.5 * (dx1 - dx_1)
            dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

            offset = dx / dxx
            keypoints[n] -= offset.reshape(-1)
        
        return keypoints