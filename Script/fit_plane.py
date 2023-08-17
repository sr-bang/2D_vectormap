from typing import List, Tuple, Callables
import copy
import numpy as np
import open3d as o3d


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    points = np.asarray(pcd.points)
    N = len(points)
    m = 3
    eta_0 = 1-confidence
    k, eps, error_star = 0, m/N, np.inf
    I = 0
    best_inliers = np.full(shape=(N,),fill_value=0.)
    best_plane = np.full(shape=(4,), fill_value=-1.)
    while pow((1-pow(eps, m)),k) >= eta_0:
        p1, p2, p3 = points[np.random.randint(N)], points[np.random.randint(N)], points[np.random.randint(N)]
        if np.linalg.norm(p1-p2) < min_sample_distance or np.linalg.norm(p2-p3) < min_sample_distance or np.linalg.norm(p1-p3) < min_sample_distance:
            continue
        n = np.cross(p2-p1,p3-p1)
        n = n/np.linalg.norm(n) 
        if n[2] < 0: # positive z direction
            n = -n
        d = -np.dot(n,p1) 
        distances = np.abs(np.dot(points, n)+d)
        error, inliers = error_func(pcd, distances, inlier_threshold)
        if error < error_star:
            I = np.sum(inliers)
            eps = I/N
            best_inliers = inliers
            error_star = error
        k = k + 1
    A = points[best_inliers]
    y = np.full(shape=(len(A),),fill_value=1.)
    best_plane[0:3] = np.linalg.lstsq(A, y, rcond=-1)[0]
    if best_plane[2] < 0:  # positive z direction
        best_plane = -best_plane
    print('best_plane',best_plane)
    return best_plane, best_inliers, k
