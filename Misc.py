import numpy as np
import open3d as o3d

def remove_statistical_outlier(in_points, nb_neighbors, std_ratio):
    pcd, _ = in_points.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def color_thresholding(pcd):
      points = np.asarray(pcd.points)
      colors = np.asarray(pcd.colors)
      white_range = np.array([[0.5, 0.5, 0.5],[1, 1, 1]])
      non_white_mask = np.all((colors < white_range[0]) | (colors > white_range[1]), axis=1)
      filtered_points = points[non_white_mask]
      filtered_colors = colors[non_white_mask]
      return filtered_points, filtered_colors

