# Different error functions for plane fitting

from typing import Tuple
import numpy as np
import open3d as o3d


def ransac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    inliers = distances < threshold
    error = np.sum(~inliers)
    return error, inliers


def msac_error(pcd: o3d.geometry.PointCloud,
               distances: np.ndarray,
               threshold: float) -> Tuple[float, np.ndarray]:
    inliers = distances < threshold
    error = np.sum((inliers==True)*pow(distances, 2) + (inliers==False)*pow(threshold, 2))
    return error, inliers


def mlesac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    sigma = threshold/2
    v = np.linalg.norm(pcd.get_max_bound()-pcd.get_min_bound())
    gamma = 1/2
    for j in range(3):
        p_i = gamma * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-pow(distances, 2) / (2 * pow(sigma, 2)))
        p_o = (1 - gamma) / v
        z_i = p_i/(p_i+p_o)
        gamma = np.sum(z_i)/len(distances)
    p_i = gamma * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-pow(distances, 2) / (2 * pow(sigma, 2)))
    p_o = (1 - gamma) / v
    error = -np.sum(np.log(p_i+p_o))
    inliers = distances < threshold
    return error, inliers
